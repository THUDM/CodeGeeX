# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PanGu predict run
"""
import json
import os
import time

import mindspore
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import moxing as mox
import numpy as np
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn.transformer import TransformerOpParallelConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.code_tokenizer import CodeTokenizer
from src.pangu_alpha import EvalNet, PanguAlphaModel
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import get_args


def load_model(args_opt):
    r"""
    The main function for load model
    """
    # Set execution mode
    context.set_context(
        save_graphs=False, mode=context.GRAPH_MODE, device_target=args_opt.device_target
    )
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            pipeline_stages=args_opt.stage_num,
        )
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path
        )
    context.set_context(
        save_graphs=False,
        save_graphs_path="/cache/graphs_of_device_id_" + str(rank),
    )
    use_past = args_opt.use_past == "true"
    print("local_rank:{}, start to run...".format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)

    parallel_config = TransformerOpParallelConfig(
        data_parallel=data_parallel_num,
        model_parallel=model_parallel_num,
        pipeline_stage=args_opt.stage_num,
        micro_batch_num=args_opt.micro_size,
        optimizer_shard=False,
        vocab_emb_dp=bool(args_opt.word_emb_dp),
        recompute=True,
    )

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        hidden_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        ffn_hidden_size=args_opt.embedding_size * 4,
        use_past=use_past,
        eod_token=args_opt.eod_id,
        eod_reset=False,
        parallel_config=parallel_config,
        load_ckpt_path=args_opt.load_ckpt_path,
        param_init_type=mstype.float32
        if args_opt.param_init_type == "fp32"
        else mstype.float16,
    )
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)
    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlphaModel(config)
    eval_net = EvalNet(pangu_alpha, pad_token=50256)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(
        np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32
    )
    current_index = Tensor(np.array([0 for _ in range(batch_size)]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(
            np.array([0 for _ in range(batch_size)]), mstype.int32
        )
        init_true = Tensor([True], mstype.bool_)
        print("Input shape:", inputs_np.shape, flush=True)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        print("is_first_iteration=True", flush=True)
        predict_layout = model_predict.infer_predict_layout(
            inputs_np, current_index, init_true, batch_valid_length
        )
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        print("is_first_iteration=False", flush=True)
        init_false = Tensor([False], mstype.bool_)
        _ = model_predict.infer_predict_layout(
            inputs_np_1, current_index, init_false, batch_valid_length
        )
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    if context.get_context("save_graphs"):
        print("==============save_graph", flush=True)
        jobid = os.environ["BATCH_JOB_ID"]
        rank_id = rank
        mox.file.make_dirs("s3://wudao-1/yyf/graphs_" + jobid)
        mox.file.copy_parallel(
            src_url="/cache/graphs_of_device_id_" + str(rank_id),
            dst_url="s3://wudao-1/yyf/graphs_" + jobid + "/" + str(rank_id),
        )
    print("======start load_distributed checkpoint", flush=True)
    if args_opt.load_ckpt_epoch > 0:
        time.sleep(rank * 0.1)
        os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}"))
        ckpt_name = f"code-13B{rank}-{args_opt.load_ckpt_epoch}.ckpt"
        if not mox.file.exists(
            os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name)
        ):
            print(f"Checkpoint from rank {rank} doesn't exist!")
        mox.file.copy(
            os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name),
            os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name),
        )
        param_dict = load_checkpoint(
            os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name)
        )
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            args_opt.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
            args_opt.has_trained_steps = int(param_dict["step_num"].data.asnumpy())
        os.mkdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1/rank_{rank}')
        while True:
            num = len(
                os.listdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1')
            )
            if num == device_num:
                break
            if rank % 8 == 0:
                print("Loaded ckpt in step 1: ", num)
            time.sleep(1)
        net_not_load = load_param_into_net(pangu_alpha, param_dict)
        print(
            "====== load_distributed checkpoint done, net_not_load: ",
            net_not_load,
            flush=True,
        )
    return model_predict, config, rank


def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(
        np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32
    )
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(
        model_predict.predict_network,
        inputs_np,
        current_index,
        init_true,
        batch_valid_length,
        file_name="pangu_alpha_1024",
        file_format="MINDIR",
    )
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(
        model_predict.predict_network,
        inputs_np_1,
        current_index,
        init_true,
        batch_valid_length,
        file_name="pangu_alpha_1",
        file_format="MINDIR",
    )
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt, rank):
    """run predict"""
    from src.generate_humaneval import generate_increment

    # Define tokenizer
    tokenizer = CodeTokenizer(mode="6b")

    # Tokenize input sentence to ids
    humaneval_path = "/home/work/sfs/xx/human_eval_x/data/humaneval_cpp.jsonl"  # TODO: set as current humaneval path
    humaneval = open(humaneval_path, "r").readlines()
    humaneval = [json.loads(task) for task in humaneval if len(task) != 0]
    samples = [task["prompt"] for task in humaneval]
    generations = []
    batch_size = config.batch_size
    verbose = rank % 8 == 0
    part = int(args_opt.part)
    gen_times = 12  # TODO: set as generation times of current task
    print(f"gen times: {gen_times}, part: {part}")
    save_path = f"/home/work/sfs/xx/pangu_alpha_code/generation_humanevalx/cpp/temp_{args_opt.temperature}/samples_{args_opt.load_ckpt_epoch}_part_{part}.jsonl"  # TODO: set as current save path
    if rank == 0 and not os.path.exists(save_path):
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        f = open(save_path, "w")
        f.close()
        os.system(f"sudo chmod 777 {save_path}")
    for i, sample in enumerate(samples):
        tag = "// language: C++\n"
        sample = tag + sample
        if rank % 8 == 0:
            print(f"=================== prompt {i} ====================")
            print(sample, flush=True)
        for j in range((gen_times + batch_size - 1) // batch_size):
            tokenized_token = tokenizer.encode_code(sample)
            input_ids = (
                np.array(tokenized_token).reshape(1, -1).repeat(batch_size, axis=0)
            )
            # Call inference
            mindspore.set_seed(j + 8 * part)
            generate_func = generate_increment
            t0 = time.perf_counter()
            output_ids = generate_func(
                model_predict, input_ids, args_opt, tokenizer, verbose
            )
            t1 = time.perf_counter()
            if rank % 8 == 0:
                print(f"=== Batch time: {t1 - t0}s")
                for k, out in enumerate(output_ids):
                    print(
                        f"=================== generation {j * batch_size + k} ===================="
                    )
                    print(out, flush=True)
                    generations.append(
                        json.dumps(
                            {"task_id": humaneval[i]["task_id"], "completion": out}
                        )
                    )
                    if rank == 0:
                        f = open(save_path, "a")
                        f.write(generations[-1] + "\n")
                        f.close()


def main():
    """Main process for predict or export model"""
    print("===Enter main!")
    opt = get_args(True)
    set_parse(opt)
    model_predict, config, rank = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict(model_predict, config, opt, rank)


if __name__ == "__main__":
    main()

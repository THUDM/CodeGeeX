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
import os
import time

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import moxing as mox
import numpy as np
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn.transformer import TransformerOpParallelConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.code_tokenizer import CodeTokenizer
from src.pangu_alpha import PanguAlphaModel, LogitsNet
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import get_args


def load_model(args_opt):
    r"""
     The main function for load model
    """
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
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
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)
    context.set_context(
        save_graphs=False,
        save_graphs_path="/cache/graphs_of_device_id_" + str(rank),
    )
    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=False,
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)

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
        if args_opt.param_init_type == 'fp32'
        else mstype.float16,
    )
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)
    ckpt_name = args_opt.load_ckpt_name
    # Define network
    pangu_alpha = PanguAlphaModel(config)
    eval_net = LogitsNet(pangu_alpha, pad_token=50256)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0 for _ in range(batch_size)]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0 for _ in range(batch_size)]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        print("Input shape:", inputs_np.shape, flush=True)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        print("is_first_iteration=True", flush=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        print("is_first_iteration=False", flush=True)
        init_false = Tensor([False], mstype.bool_)
        _ = model_predict.infer_predict_layout(inputs_np_1, init_false, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    if context.get_context("save_graphs"):
        print("==============save_graph", flush=True)
        jobid = os.environ["BATCH_JOB_ID"]
        rank_id = rank
        mox.file.make_dirs("s3://wudao-1/yyf/graphs_" + jobid)
        mox.file.copy_parallel(src_url="/cache/graphs_of_device_id_" + str(rank_id),
                               dst_url="s3://wudao-1/yyf/graphs_" + jobid + "/" + str(rank_id))
    print("======start load_distributed checkpoint", flush=True)
    if args_opt.load_ckpt_epoch > 0:
        time.sleep(rank * 0.1)
        os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}"))
        ckpt_name = f"code-13B{rank}-{args_opt.load_ckpt_epoch}.ckpt"
        if not mox.file.exists(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name)):
            print(f"Checkpoint from rank {rank} doesn't exist!")
        mox.file.copy(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name),
                      os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name))
        param_dict = load_checkpoint(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name))
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            args_opt.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
            args_opt.has_trained_steps = int(param_dict["step_num"].data.asnumpy())
        os.mkdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1/rank_{rank}')
        while True:
            num = len(os.listdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1'))
            if num == device_num:
                break
            if rank % 8 == 0:
                print("Loaded ckpt in step 1: ", num)
            time.sleep(1)
        net_not_load = load_param_into_net(pangu_alpha, param_dict)
        print("====== load_distributed checkpoint done, net_not_load: ", net_not_load, flush=True)
    return model_predict, config, rank


def run_predict(model_predict, config, args_opt, rank):
    """run predict"""
    # Define tokenizer
    tokenizer = CodeTokenizer(mode='6b')

    # Tokenize input sentence to ids
    batch_size = config.batch_size
    input_ids = np.array(
        [8189, 11059, 198, 29584, 25, 198, 11377, 1398, 28186, 1391, 198, 50268, 11377, 9037, 25131, 468, 26125, 36,
         3639, 7, 600, 21737, 997, 82, 11, 493, 11387, 8, 1391, 198, 50272, 1640, 357, 600, 1312, 796, 657, 26, 1312,
         1279, 997, 82, 13, 13664, 532, 352, 26, 1312, 29577, 1391, 198, 50276, 1640, 357, 600, 474, 796, 1312, 1343,
         352, 26, 474, 1279, 997, 82, 13, 13664, 26, 474, 29577, 1391, 198, 50280, 361, 357, 37372, 13, 8937, 7, 77,
         5700, 58, 72, 60, 532, 997, 82, 58, 73, 12962, 1279, 11387, 8, 1391, 198, 50284, 7783, 2081, 26, 198, 50280,
         92, 198, 50276, 92, 198, 50272, 92, 198, 50272, 7783, 3991, 26, 198, 50268, 92, 198, 92, 198, 5247, 25, 198],
        dtype=np.int32)
    valid_length = input_ids.shape[0]
    input_ids = np.concatenate((input_ids, np.ones(2048 - valid_length, dtype=np.int32) * 50256))
    attention_mask = np.tril(np.ones((2048, 2048)))
    attention_mask[valid_length:] = 0
    input_ids = input_ids.reshape(1, -1).repeat(config.batch_size, axis=0)
    current_index = valid_length - 1 if valid_length - 1 > 0 else 0
    init = Tensor([False], mstype.bool_)
    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    batch_valid_length = Tensor(np.array([current_index for _ in range(batch_size)]), mstype.int32)
    output_logits = model_predict.predict(Tensor(input_ids, mstype.int32),
                                          init, batch_valid_length)
    output = output_logits.asnumpy()
    if rank == 0:
        np.save("/home/work/sfs/xx/pangu_alpha_code/output_6_7375_8.13.npy", output)  # TODO: set as current save path
        os.system(
            "chmod 777 /home/work/sfs/xx/pangu_alpha_code/output_6_7375_8.13.npy")  # TODO: set as current save path
        print("== Output shape: ", output.shape)


def main():
    """Main process for predict or export model"""
    print("===Enter main!")
    opt = get_args(True)
    set_parse(opt)
    model_predict, config, rank = load_model(opt)
    run_predict(model_predict, config, opt, rank)


if __name__ == "__main__":
    main()

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
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn.transformer import TransformerOpParallelConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.code_tokenizer import CodeTokenizer
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.pangu_alpha_fp16_predict import EvalNet, PanguAlphaModel
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
    print("===args_opt: ", args_opt, flush=True)
    print("===device_num is: ", device_num, flush=True)
    args_opt.op_level_model_parallel_num = 1
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    print("===data_parallel_num is: ", data_parallel_num, flush=True)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=False,
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1
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
    eval_net = EvalNet(pangu_alpha, pad_token=50256)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        print("Input shape:", inputs_np.shape, flush=True)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        print("is_first_iteration=True", flush=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        print("is_first_iteration=False", flush=True)
        init_false = Tensor([False], mstype.bool_)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_false, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    print("======start load_distributed checkpoint", flush=True)

    time.sleep(rank * 0.5)
    if not os.path.exists(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}")):
        os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}"))
    ckpt_name = f"code-13B0-50.ckpt"  # TODO: set to current ckpt name
    if not mox.file.exists(os.path.join(args_opt.load_ckpt_path, "rank_0", ckpt_name)):
        print(f"Checkpoint from rank {rank} doesn't exist!")
    mox.file.copy(os.path.join(args_opt.load_ckpt_path, "rank_0", ckpt_name),
                  os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name))
    param_dict = load_checkpoint(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}", ckpt_name))
    # TODO: add them back if not for the 1st run!
    if param_dict.get("epoch_num") and param_dict.get("step_num"):
        args_opt.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        args_opt.has_trained_steps = int(param_dict["step_num"].data.asnumpy())
    if not os.path.exists(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1/rank_{rank}'):
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
    if not os.path.exists("/home/work/sfs/cache/ckpts_npy/"):
        os.mkdir("/home/work/sfs/cache/ckpts_npy/")
    for k, weight in pangu_alpha.parameters_dict().items():
        print(k)
        np.save(os.path.join("/home/work/sfs/cache/ckpts_npy/", f"{k}.npy"), weight.asnumpy())
    rank_obs_save_path = "./"  # TODO: set to current obs path for saving
    if not mox.file.exists(rank_obs_save_path):
        mox.file.make_dirs(rank_obs_save_path)
    mox.file.copy_parallel("/home/work/sfs/cache/ckpts_npy/", rank_obs_save_path)
    print("======= npy saved")
    return model_predict, config, rank


def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt, rank):
    """run predict"""
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = CodeTokenizer(mode='6b')

    # Tokenize input sentence to ids
    samples = [
        "# language: Python\ndef add(a, b):\n    '''\n    Find the sum of a and b.\n    '''\n",
        "def add(a, b):\n    '''\n    Find the sum of a and b.\n    '''\n",
        "# language: Python\ndef optimization():\n    '''\n    Find the maximum of P=E**2*R/(R + r)**2 if E and r are fixed but R varies. Import sympy. Use sympy. Find where the derivative is equal to zero. Substitute the value of R into P.\n    '''\n",
        "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "// language: C++\nint add(int a, int b) {\n    /* Find the sum of a and b. */\n",
        "int add(int a, int b) {\n    /* Find the sum of a and b. */\n",
        "bool prime(int n) {\n    // Find whether n is a prime number\n",
        "// language: JavaScript\nfunction add(a, b) {\n    // Find the sum of a and b.\n",
        "# language: R\nadd<-function(a, b) {\n    # Find the sum of a and b.\n",
    ]
    verbose = False
    for i, sample in enumerate(samples):
        for _ in range(1):
            tokenized_token = tokenizer.encode_code(sample)
            input_ids = np.array(tokenized_token).reshape(1, -1)
            # Call inference
            generate_func = generate_increment if config.use_past else generate
            t0 = time.perf_counter()
            output_ids = generate_func(model_predict, input_ids, args_opt, verbose)
            # Decode output ids to sentence
            t1 = time.perf_counter()
            output_samples = tokenizer.decode_code(output_ids.tolist())
            output_samples_str = "".join(output_samples)
            if rank % 8 == 0:
                print(f"=================== prompt {i} ====================")
                print(sample, flush=True)
                print(f"=================== generation {i} ====================")
                print(output_samples_str, flush=True)
                print(
                    f"=== Total time (s): {t1 - t0}, {output_ids.shape[-1] - input_ids.shape[-1]} tokens, {(output_ids.shape[-1] - input_ids.shape[-1]) / (t1 - t0)} token/s")


def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config, rank = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict(model_predict, config, opt, rank)


if __name__ == "__main__":
    main()

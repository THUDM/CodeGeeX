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
PanguAlpha train script
"""

import datetime
import argparse
import numpy as np
import json
import glob
import os
import math
import time
from pathlib2 import Path
from collections import defaultdict
import moxing as mox

from tensorboardX import SummaryWriter

from mindspore import context
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn import TransformerOpParallelConfig, CrossEntropyLoss
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

import mindspore
from mindspore.train.serialization import load_checkpoint, build_searched_strategy, save_checkpoint, \
    merge_sliced_parameter, _convert_to_list, _convert_to_layout
from mindspore.common.parameter import Parameter
from mindspore.train.node_strategy_pb2 import ParallelStrategyMap, ParallelLayouts
from mindspore import Tensor
from mindspore.parallel._utils import _infer_rank_list, _remove_repeated_slices

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import LossCallBack, SaveCheckpointCallback
from mindspore.profiler import Profiler

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {"params": decay_params, "weight_decay": 1e-1},
        {"params": other_params, "weight_decay": 0.0},
        {"order_params": params},
    ]
    return group_params


def add_checkpoint_callback_policy(args_param, callback, rank_id):
    r"""
    Add checkpoint policy to callback.
    """
    if args_param.save_checkpoint:
        # checkpoint store epoch_num and step_num info
        ckpt_append_info = [{"epoch_num": args_param.has_trained_epoches, "step_num": args_param.has_trained_steps}]
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=args_param.save_checkpoint_steps,
            keep_checkpoint_max=args_param.keep_checkpoint_max,
            integrated_save=False,
            append_info=ckpt_append_info,
        )

        # save checkpoint into rank directory
        ckpoint_cb = ModelCheckpoint(prefix=args_param.ckpt_name_prefix + str(rank_id),
                                     directory=os.path.join(args_param.save_checkpoint_path, f"rank_{rank_id}"),
                                     config=ckpt_config)

        callback.append(ckpoint_cb)

        saveckpt_cb = SaveCheckpointCallback(cache_dir=args_param.save_checkpoint_path,
                                             bucket=args_param.save_checkpoint_obs_path,
                                             local_rank=rank_id,
                                             has_trained_epoch=args_param.has_trained_epoches,
                                             has_trained_step=args_param.has_trained_steps,
                                             syn_times=args_param.save_checkpoint_steps)
        callback.append(saveckpt_cb)


def set_parallel_context(args_opt):
    r"""Set parallel context"""
    D.init()
    device_num = D.get_group_size()
    rank = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
        full_batch=bool(args_opt.full_batch), strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
        enable_parallel_optimizer=bool(args_opt.optimizer_shard), strategy_ckpt_save_file='strategy.ckpt',
        optimizer_weight_shard_size=16, optimizer_weight_shard_aggregated_save=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank, device_num


def download_ckpt(args_opt, file_num, rank_num, rank_id):
    ckpt_list = []
    for rank in range(0, file_num):
        ckpt_name = f"code-13B{rank}-{args_opt.load_ckpt_epoch}.ckpt"
        local_file = os.path.join(args_opt.save_checkpoint_path, f"origin_rank_{rank}", ckpt_name)
        ckpt_list.append(local_file)
        if rank % rank_num != rank_id:
            continue
        time.sleep(rank * 0.05)
        if not os.path.exists(os.path.join(args_opt.save_checkpoint_path, f"origin_rank_{rank}")):
            os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"origin_rank_{rank}"))
        if not mox.file.exists(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name)):
            print(f"Checkpoint from rank {rank} doesn't exist!")
        mox.file.copy(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name), local_file)
        print("===download ckpt ok: ", local_file, flush=True)
    print(ckpt_list)
    return ckpt_list


def get_needed_model_parallel_list(train_strategy_file, self_rank):
    train_strategy_origin = build_searched_strategy(train_strategy_file)
    strategy_keys = list(train_strategy_origin.keys())
    train_strategy = _convert_to_list(train_strategy_origin)
    rank_list = _infer_rank_list(train_strategy, None)
    needed_ckpt_ranks = []
    for param_name in strategy_keys:
        param_needs_rank_list = rank_list[param_name][0]
        if len(param_needs_rank_list) > len(needed_ckpt_ranks):  # 实际上应该求并集。
            needed_ckpt_ranks = param_needs_rank_list
    return needed_ckpt_ranks


def transform_model_parallel(restore_local_ckpt_file_list, train_strategy_file, save_path, using_fp16=False):
    # check whether the ckpt_file has been download
    for local_file in restore_local_ckpt_file_list:
        if not os.path.exists(local_file):
            raise ValueError("ckpt not download: ", restore_local_ckpt_file_list)
    time.sleep(0.1)
    param_total_dict = defaultdict(dict)
    for file_index, local_file in enumerate(restore_local_ckpt_file_list):
        param_dict = load_checkpoint(local_file)
        for param_name, param in param_dict.items():
            if "adam" in param_name:
                continue
            print(f"===loading {file_index}: ", param_name, flush=True)
            param_total_dict[param_name][file_index] = param
    print("===load param done.", flush=True)
    train_strategy_origin = build_searched_strategy(train_strategy_file)
    train_strategy = _convert_to_list(train_strategy_origin)
    rank_list = _infer_rank_list(train_strategy, None)
    strategy_keys = list(train_strategy_origin.keys())
    merged_param_list = []
    for param_name in param_total_dict.keys():
        if "adam" in param_name:
            continue
        if param_name not in strategy_keys:
            each_param = {"name": param_name}
            each_param["data"] = param_total_dict[param_name][0]
            print("====", param_name, param_total_dict[param_name][0].data.asnumpy().shape, flush=True)
            merged_param_list.append(each_param)
            continue
        param_unique_strategy = _remove_repeated_slices(train_strategy[param_name])
        _param_unique_strategy = _convert_to_layout(param_name, param_unique_strategy)
        sliced_params = []
        if using_fp16 and "embedding" not in param_name and "layernorm" not in param_name:
            for i in rank_list[param_name][0]:
                slice_param = param_total_dict[param_name][i]
                layerwise_parallel = slice_param.layerwise_parallel
                requires_grad = slice_param.requires_grad
                sliced_data = sliced_params.data.asnumpy()
                sliced_data = sliced_data.astype(np.float16)
                paramete_fp16 = Parameter(Tensor(sliced_data), param_name, requires_grad, layerwise_parallel)
                sliced_params.append(paramete_fp16)
        else:
            sliced_params = [param_total_dict[param_name][i] for i in rank_list[param_name][0]]
        merged_param = merge_sliced_parameter(sliced_params, _param_unique_strategy)
        each_param = {"name": param_name}
        each_param["data"] = merged_param
        print("====", param_name, merged_param.data.asnumpy().shape, flush=True)
        merged_param_list.append(each_param)
    save_file = os.path.join(save_path, "predict_1p.ckpt")
    save_checkpoint(merged_param_list, save_file)
    return save_file


def run_transform_model_parallel_ckpt(args_opt):
    # Set execution mode
    context.set_context(
        mode=context.GRAPH_MODE, device_target=args_opt.device_target
    )
    # Set parallel context
    rank = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank, device_num = set_parallel_context(args_opt)
    print("=====rank is: ", rank, flush=True)
    ckpt_file_list = download_ckpt(args_opt, 8, device_num, rank)
    if rank != 0:
        return
    needed_ckpt_ranks = get_needed_model_parallel_list(args_opt.strategy_load_ckpt_path, rank)
    restore_local_ckpt_file_list = [ckpt_file_list[i] for i in needed_ckpt_ranks]
    print("====restore_local_ckpt_file_list====", restore_local_ckpt_file_list, flush=True)
    save_path = os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file = transform_model_parallel(restore_local_ckpt_file_list, args_opt.strategy_load_ckpt_path, save_path)
    obs_save_path = args_opt.save_checkpoint_obs_path
    time.sleep(rank * 0.1)
    if not mox.file.exists(obs_save_path):
        mox.file.make_dirs(obs_save_path)
    rank_obs_save_path = os.path.join(obs_save_path, f"rank_{rank}")
    if not mox.file.exists(rank_obs_save_path):
        mox.file.make_dirs(rank_obs_save_path)
    rank_obs_save_file = os.path.join(rank_obs_save_path, f"code-13B{rank}-{args_opt.load_ckpt_epoch}.ckpt")
    if not os.path.exists(save_file):
        raise ValueError(save_file + " not exists")
    mox.file.copy(save_file, rank_obs_save_file)
    print("=====save ok, save_path", save_path)


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    run_transform_model_parallel_ckpt(opt)

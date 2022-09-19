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
import numpy as np
import glob
import os
import math
import time
from collections import defaultdict
import moxing as mox

from tensorboardX import SummaryWriter

from mindspore import context
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net

import mindspore
from mindspore.train.serialization import load_checkpoint, build_searched_strategy, save_checkpoint, \
    merge_sliced_parameter
from mindspore.common.parameter import Parameter
from mindspore import Tensor

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from mindspore.profiler import Profiler

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


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
        ckpt_name = f"code-13B{rank}_22-{args_opt.load_ckpt_epoch}_2.ckpt"
        local_file = os.path.join(args_opt.save_checkpoint_path, f"origin_rank_{rank}", ckpt_name)
        ckpt_list.append(local_file)
        if rank % rank_num != rank_id:
            continue
        time.sleep(rank * 0.05)
        os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"origin_rank_{rank}"))
        if not mox.file.exists(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name)):
            print(f"Checkpoint from rank {rank} doesn't exist!")
        mox.file.copy(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name), local_file)
        print("===download ckpt ok: ", local_file, flush=True)
    return ckpt_list


def get_needed_opt_shard_list(train_strategy_file, self_rank):
    train_strategy_origin = build_searched_strategy(train_strategy_file)
    strategy_keys = list(train_strategy_origin.keys())
    needed_ckpt_ranks = []
    for param_name in strategy_keys:
        opt_weight_shard_size = train_strategy_origin[param_name].opt_weight_shard_size
        opt_weight_shard_step = train_strategy_origin[param_name].opt_weight_shard_step
        if opt_weight_shard_size <= 0:
            continue
        group_index = self_rank % opt_weight_shard_step
        current_needed_ckpt_ranks = [group_index + i * opt_weight_shard_step for i in range(0, opt_weight_shard_size)]
        if len(current_needed_ckpt_ranks) > len(needed_ckpt_ranks):
            needed_ckpt_ranks = current_needed_ckpt_ranks
    return needed_ckpt_ranks


def transform_opt_shard(restore_local_ckpt_file_list, train_strategy_file, save_path):
    # check whether the ckpt_file has been download
    for local_file in restore_local_ckpt_file_list:
        if not os.path.exists(local_file):
            raise ValueError("ckpt not download: ", restore_local_ckpt_file_list)
    time.sleep(0.1)
    param_total_dict = defaultdict(dict)
    for file_index, local_file in enumerate(restore_local_ckpt_file_list):
        param_dict = load_checkpoint(local_file)
        for param_name, param in param_dict.items():
            param_total_dict[param_name][file_index] = param

    train_strategy_origin = build_searched_strategy(train_strategy_file)
    strategy_keys = list(train_strategy_origin.keys())
    merged_param_list = []
    for param_name in param_total_dict.keys():
        if param_name not in strategy_keys:
            each_param = {"name": param_name}
            each_param["data"] = param_total_dict[param_name][0]
            print("====", param_name, param_total_dict[param_name][0].data.asnumpy().shape, flush=True)
            merged_param_list.append(each_param)
            continue
        opt_weight_shard_size = train_strategy_origin[param_name].opt_weight_shard_size
        opt_weight_shard_step = train_strategy_origin[param_name].opt_weight_shard_step
        if opt_weight_shard_step == 0:
            print("====not opt shard:", param_name)
            each_param = {"name": param_name}
            each_param["data"] = param_total_dict[param_name][0]
            print("====", param_name, param_total_dict[param_name][0].data.asnumpy().shape, flush=True)
            merged_param_list.append(each_param)
            continue
        print("====do opt shard:", param_name)
        sliced_params = [param_total_dict[param_name][i] for i in range(len(param_total_dict[param_name]))]
        merged_param = merge_sliced_parameter(sliced_params, None)
        each_param = {"name": param_name}
        each_param["data"] = merged_param
        print("====", param_name, merged_param.data.asnumpy().shape, flush=True)
        merged_param_list.append(each_param)
    save_file = os.path.join(save_path, "predict.ckpt")
    save_checkpoint(merged_param_list, save_file)
    return save_file


def run_transform_opt_shard_ckpt(args_opt):
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
    ckpt_file_list = download_ckpt(args_opt, 128, device_num, rank)
    needed_ckpt_ranks = get_needed_opt_shard_list(args_opt.strategy_load_ckpt_path, rank)
    restore_local_ckpt_file_list = [ckpt_file_list[i] for i in needed_ckpt_ranks]
    print("====restore_local_ckpt_file_list====", restore_local_ckpt_file_list, flush=True)
    save_path = os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}")
    os.mkdir(save_path)
    save_file = transform_opt_shard(restore_local_ckpt_file_list, args_opt.strategy_load_ckpt_path, save_path)
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
    run_transform_opt_shard_ckpt(opt)

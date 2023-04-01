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
import math
import os
import time

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
import moxing as mox
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn import TransformerOpParallelConfig, CrossEntropyLoss
from mindspore.profiler import Profiler
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import (
    load_distributed_checkpoint,
    load_checkpoint,
    load_param_into_net,
)
from tensorboardX import SummaryWriter

from src.adam import AdamWeightDecayOp
from src.callbacks import EvalCallBack, LossCallBack, SaveCheckpointCallback
from src.dataset_finetune import create_dataset
from src.metrics import PPLMetric, ValidationLoss
from src.pangu_alpha import PanguAlphaModel, PanGUAlphaWithFinetuneLoss
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.pangu_alpha_wrapcell_finetune import PanguAlphaTrainOneStepWithLossScaleCell
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".."
)
print("project_root:", project_root)


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = (
        lambda x: "layernorm" not in x.name.lower() and "bias" not in x.name.lower()
    )
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
        ckpt_append_info = [
            {
                "epoch_num": args_param.has_trained_epoches,
                "step_num": args_param.has_trained_steps,
            }
        ]
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=args_param.save_checkpoint_steps,
            keep_checkpoint_max=args_param.keep_checkpoint_max,
            integrated_save=False,
            append_info=ckpt_append_info,
        )

        # save checkpoint into rank directory
        ckpoint_cb = ModelCheckpoint(
            prefix=args_param.ckpt_name_prefix + str(rank_id),
            directory=os.path.join(args_param.save_checkpoint_path, f"rank_{rank_id}"),
            config=ckpt_config,
        )

        callback.append(ckpoint_cb)

        saveckpt_cb = SaveCheckpointCallback(
            cache_dir=args_param.save_checkpoint_path,
            bucket=args_param.save_checkpoint_obs_path,
            local_rank=rank_id,
            has_trained_epoch=args_param.has_trained_epoches,
            has_trained_step=args_param.has_trained_steps,
            syn_times=args_param.save_checkpoint_steps,
        )
        callback.append(saveckpt_cb)


def set_parallel_context(args_opt):
    r"""Set parallel context"""
    D.init()
    device_num = D.get_group_size()
    rank = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank, device_num))
    if device_num < 128:
        args_opt.optimizer_shard = 0
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
        gradients_mean=False,
        full_batch=bool(args_opt.full_batch),
        strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
        enable_parallel_optimizer=bool(args_opt.optimizer_shard),
        strategy_ckpt_save_file="strategy.ckpt",
    )
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank, device_num


def run_train(args_opt):
    r"""The main training process."""
    os.environ["HCCL_CONNECT_TIMEOUT"] = "2000"
    # Set execution mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.profiling:
        profiler = Profiler(output_path="/cache/profiler_data")
    context.set_context(variable_memory_max_size="30GB")
    # Set parallel context
    rank = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank, device_num = set_parallel_context(args_opt)
    context.set_context(
        save_graphs=False,
        save_graphs_path="/cache/graphs_of_device_id_" + str(rank),
    )

    # copy data from the cloud to the /cache/Data
    cache_url = "/cache/Data/"
    eval_cache_url = "/cache/EvalData/"
    if not args_opt.offline:
        download_data(
            src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank
        )
        download_data(
            src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank
        )
    # Set model property
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    batch_size = args_opt.per_batch_size * data_parallel_num
    parallel_config = TransformerOpParallelConfig(
        data_parallel=data_parallel_num,
        model_parallel=model_parallel_num,
        pipeline_stage=args_opt.stage_num,
        micro_batch_num=args_opt.micro_size,
        optimizer_shard=bool(args_opt.optimizer_shard),
        vocab_emb_dp=bool(args_opt.word_emb_dp),
        recompute=True,
        gradient_aggregation_group=args_opt.gradient_aggregation_group,
    )

    micro_interleaved_size = args_opt.micro_interleaved_size
    config = PanguAlphaConfig(
        batch_size=batch_size // micro_interleaved_size,
        num_heads=args_opt.num_heads,
        hidden_size=args_opt.embedding_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        num_layers=args_opt.num_layers,
        ffn_hidden_size=args_opt.embedding_size * 4,
        eod_token=args_opt.eod_id,
        load_ckpt_path=args_opt.load_ckpt_path,
        param_init_type=mstype.float32
        if args_opt.param_init_type == "fp32"
        else mstype.float16,
        dropout_rate=args_opt.dropout_rate,
        enable_offload=bool(args_opt.opt_offload),
        use_moe=bool(args_opt.use_moe),
        per_dp_dim_expert_num=args_opt.per_dp_dim_expert_num,
        hidden_act="fast_gelu" if args_opt.device_target != "GPU" else "gelu",
        parallel_config=parallel_config,
    )
    print("===config is: ", config, flush=True)
    # Define network
    pangu_alpha = PanguAlphaModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    if micro_interleaved_size > 1:
        print("===using MicroBatchInterleaved", flush=True)
        pangu_alpha_with_loss_net = MicroBatchInterleaved(
            PanGUAlphaWithFinetuneLoss(config, pangu_alpha, loss),
            micro_interleaved_size,
        )
    else:
        pangu_alpha_with_loss_net = PanGUAlphaWithFinetuneLoss(
            config, pangu_alpha, loss
        )
    pangu_alpha_with_loss = _VirtualDatasetCell(pangu_alpha_with_loss_net)
    print("=====args_opt is: ", args_opt, flush=True)
    # Warm-up and cosine decay learning rate
    lr = LearningRate(
        learning_rate=args_opt.start_lr,
        end_learning_rate=args_opt.end_lr,
        warmup_steps=args_opt.warmup_step,
        decay_steps=args_opt.decay_steps,
    )
    params = pangu_alpha_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(
            group_params,
            learning_rate=lr,
            eps=1e-8,
            beta1=0.9,
            beta2=0.95,
            param_init_type=config.param_init_type,
        )
    else:
        optimizer = FP32StateAdamWeightDecay(
            group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95
        )
    # Initial scaling sens
    loss_scale_value = math.pow(2, 32)
    epoch_num = args_opt.epoch_size

    if args_opt.load_ckpt_epoch > 0:
        time.sleep(rank * 0.05)
        os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}"))
        ckpt_name = f"code-13B{rank}_21-{args_opt.load_ckpt_epoch}_2.ckpt"
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
        # TODO: remove after warming-up!
        param_dict.pop("global_step")
        # TODO: add them back if not for the 1st run!
        # if param_dict.get("epoch_num") and param_dict.get("step_num"):
        #     args_opt.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        #     args_opt.has_trained_steps = int(param_dict["step_num"].data.asnumpy())
        # args_opt.has_trained_steps = 9000

        os.mkdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1/rank_{rank}')
        while True:
            num = len(
                os.listdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1')
            )
            if num == device_num:
                break
            if rank % 64 == 0:
                print("Loaded ckpt in step 1: ", num)
            time.sleep(1)

    # if args_opt.tb_dir is not None and rank == device_num - 1:
    if args_opt.tb_dir is not None and rank == 0:
        os.makedirs(args_opt.tb_dir, exist_ok=True)
        summary_writer = SummaryWriter(args_opt.tb_dir)
        os.system(f"chmod 777 -R {args_opt.tb_dir}")
    else:
        summary_writer = None

    # Dataset loading mindrecord files
    ds, ds_eval = create_dataset(
        config.batch_size * micro_interleaved_size,
        data_path=args_opt.code_data,
        args_opt=args_opt,
        data_start_index=0,
        eod_reset=config.eod_reset,
        full_batch=bool(args_opt.full_batch),
        eod_id=args_opt.eod_id,
        device_num=device_num,
        rank=rank,
        epoch=epoch_num,
        train_and_eval=bool(args_opt.train_and_eval_mode),
        val_ratio=0.001,
    )
    actual_epoch_num = int(ds.get_dataset_size() / args_opt.sink_size)
    callback = [
        TimeMonitor(args_opt.sink_size),
    ]
    update_cell = DynamicLossScaleUpdateCell(
        loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000
    )
    pangu_alpha_with_grads = PanguAlphaTrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss,
        optimizer=optimizer,
        scale_update_cell=update_cell,
        enable_global_norm=True,
        config=config,
    )
    if ds_eval:
        ppl_metric = PPLMetric(config.seq_length)
        validation_loss = ValidationLoss(config.seq_length)
        model = Model(
            pangu_alpha_with_grads,
            eval_network=pangu_alpha_with_loss,
            metrics={"ppl": ppl_metric, "validation_loss": validation_loss},
        )
        callback.append(
            EvalCallBack(
                model=model,
                eval_dataset=ds_eval,
                ppl_metric=ppl_metric,
                validation_loss=validation_loss,
                print_per_step=20,
                has_trained_step=args_opt.has_trained_steps,
                local_rank=rank,
                rank_size=device_num,
                tb_writer=summary_writer,
            )
        )
    else:
        model = Model(pangu_alpha_with_grads)
    if args_opt.load_ckpt_epoch > 0:
        print("===build model and load ckpt")
        time_stamp = datetime.datetime.now()
        print(
            f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} before building",
            flush=True,
        )
        model.build(
            train_dataset=ds, sink_size=args_opt.sink_size, epoch=actual_epoch_num
        )
        time_stamp = datetime.datetime.now()
        print(
            f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} before loading ckpt",
            flush=True,
        )
        load_param_into_net(pangu_alpha_with_loss, param_dict)

        os.mkdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/2/rank_{rank}')
        while True:
            num = len(
                os.listdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/2')
            )
            if num == device_num:
                break
            if rank % 64 == 0:
                print("Loaded ckpt in step 2: ", num)
            time.sleep(1)

    callback.append(
        LossCallBack(
            name=args_opt.ckpt_name_prefix,
            dataset_size=args_opt.sink_size,
            local_rank=rank,
            rank_size=device_num,
            has_trained_epoch=args_opt.has_trained_epoches,
            has_trained_step=args_opt.has_trained_steps,
            micro_size=args_opt.micro_size * micro_interleaved_size,
            tb_writer=summary_writer,
        )
    )
    if not args_opt.profiling:
        add_checkpoint_callback_policy(args_opt, callback, rank)
    if args_opt.incremental_training:
        strategy = model.infer_train_layout(
            train_dataset=ds, sink_size=args_opt.sink_size
        )
        print("======start load_distributed checkpoint", flush=True)
        # For 2.6B and 13B models, the number of ckpt files is 512.
        ckpt_file_list = [
            os.path.join(args_opt.load_ckpt_path, f"filerted_{ckpt_rank}.ckpt")
            for ckpt_rank in range(0, 512)
        ]
        print(f"Loading from path {ckpt_file_list[0]}", flush=True)
        load_distributed_checkpoint(model.train_network, ckpt_file_list, strategy)
    print(
        "Dataset size: {}, actual_epoch_num: {}".format(
            ds.get_dataset_size(), actual_epoch_num
        ),
        flush=True,
    )

    try:
        model.train(
            10 if args_opt.profiling else actual_epoch_num,
            ds,
            callbacks=callback,
            sink_size=args_opt.sink_size,
            dataset_sink_mode=True,
        )
    finally:
        if args_opt.profiling:
            jobid = os.environ["BATCH_JOB_ID"]
            profiler.analyse()
            rank_id = rank
            if context.get_context("save_graphs"):
                mox.file.make_dirs("s3://wudao-1/yyf/graphs_" + jobid)
                mox.file.copy_parallel(
                    src_url="/cache/graphs_of_device_id_" + str(rank_id),
                    dst_url="s3://wudao-1/yyf/graphs_" + jobid + "/" + str(rank_id),
                )
            if rank_id % 8 == 0:
                mox.file.make_dirs("s3://wudao-1/yyf/profiler_" + jobid)
                mox.file.copy_parallel(
                    src_url="/cache/profiler_data",
                    dst_url="s3://wudao-1/yyf/profiler_" + jobid + "/" + str(rank_id),
                )


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    if opt.stage_num > 1:
        if bool(opt.use_moe) or bool(opt.opt_offload):
            raise ValueError(
                "Currently, moe and host device mode is not supported in pipeline parallel."
            )
    else:
        run_train(opt)

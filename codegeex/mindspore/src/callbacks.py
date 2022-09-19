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
Callbacks
"""

import math
import os
import time
from multiprocessing import Process

import moxing as mox
from mindspore import context
from mindspore.communication.management import get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(
            self,
            name,
            dataset_size=-1,
            local_rank=0,
            rank_size=1,
            has_trained_epoch=0,
            has_trained_step=0,
            micro_size=1,
            sink_size=2,
            tb_writer=None,
    ):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.rank_size = rank_size
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step
        self.micro_size = micro_size
        self.sink_size = sink_size

        self.summary_writer = tb_writer
        print("load has trained epoch :{} and step: {}".format(has_trained_epoch, has_trained_step), flush=True)

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                epoch_num -= 1
            date = time.asctime(time.localtime(time.time()))
            loss_value = cb_params.net_outputs[0].asnumpy() / self.micro_size

            if self.summary_writer is not None:
                print(f"writing: {loss_value.item()}, {cb_params.net_outputs[2].asnumpy()}")
                self.summary_writer.add_scalar(
                    tag="training_loss",
                    scalar_value=loss_value.item(),
                    global_step=cb_params.cur_step_num
                                + int(self.has_trained_step),
                )
                self.summary_writer.add_scalar(
                    tag="loss_scale",
                    scalar_value=cb_params.net_outputs[2].asnumpy(),
                    global_step=cb_params.cur_step_num
                                + int(self.has_trained_step),
                )
            print(
                f"time: {date} local_rank: {int(self.local_rank)}, epoch: {int(epoch_num) + int(self.has_trained_epoch)}, step: {cb_params.cur_step_num + int(self.has_trained_step)}, output is {loss_value}, overflow is {cb_params.net_outputs[1].asnumpy()}, scale is {cb_params.net_outputs[2].asnumpy()}")


class EvalCallBack(Callback):
    """
    Monitor the ppl loss in evaluating.
    Note:
        If per_print_times is 0, do NOT print loss.

    Args:
        print_per_step (int): Print loss every times. Default: 1.
    """

    def __init__(self, model, eval_dataset, ppl_metric, validation_loss, print_per_step=250, has_trained_step=0,
                 local_rank=0, rank_size=1, tb_writer=None, lang=None):
        super(EvalCallBack, self).__init__()
        if not isinstance(print_per_step, int) or print_per_step < 0:
            raise ValueError("print_per_step must be int and >= 0.")
        self.print_per_step = print_per_step
        self.model = model
        self.eval_dataset = eval_dataset
        self.pplMetric = ppl_metric
        self.validation_loss = validation_loss
        self.has_trained_step = has_trained_step
        self.local_rank = local_rank
        self.rank_size = rank_size
        self.pplMetric.clear()
        self.validation_loss.clear()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.strategy_ckpt_save_file = context.get_auto_parallel_context("strategy_ckpt_save_file")
        self.strategy_ckpt_load_file = context.get_auto_parallel_context("strategy_ckpt_load_file")
        self.summary_writer = tb_writer
        self.lang = lang

    def step_end(self, run_context):
        """
        step end
        """
        cb_params = run_context.original_args()
        current_step = cb_params.cur_step_num + self.has_trained_step
        if current_step % self.print_per_step != 0:
            return
        self.pplMetric.clear()
        self.validation_loss.clear()
        if self.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            context.set_auto_parallel_context(strategy_ckpt_save_file="",
                                              strategy_ckpt_load_file=self.strategy_ckpt_save_file)
        rank_id = 0
        if self.parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL,
                                  ParallelMode.AUTO_PARALLEL, ParallelMode.DATA_PARALLEL):
            rank_id = get_rank()
        print("validation begin")
        start_time = time.time()
        out = self.model.eval(self.eval_dataset)
        end_time = time.time()
        eval_time = int(end_time - start_time)

        time_str = time.strftime("%Y-%m-%d %H:%M%S", time.localtime())
        out_str = f"{time_str} == Language {self.lang}; Rank {rank_id} Eval: {out}; eval_time: {eval_time}s"
        print(out_str)
        if self.summary_writer is not None:
            print(f"writing: {out}")
            tag = "validation_loss" if self.lang is None else f"validaton_loss/{self.lang}"
            self.summary_writer.add_scalar(
                tag=tag,
                scalar_value=out['ppl'],
                global_step=cb_params.cur_step_num + int(self.has_trained_step),
            )
        context.set_auto_parallel_context(strategy_ckpt_save_file=self.strategy_ckpt_save_file,
                                          strategy_ckpt_load_file=self.strategy_ckpt_load_file)


class SaveCheckpointCallback(Callback):
    def __init__(self, cache_dir, bucket, local_rank=0, has_trained_epoch=0, has_trained_step=0, syn_times=100):
        self.cache_dir = os.path.join(cache_dir, f"rank_{local_rank}")
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = os.path.join(bucket, f"rank_{local_rank}")
        self.syn_times = syn_times

        if not mox.file.exists(self.bucket):
            print("Creating checkpoint bucket dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        if cur_step % self.syn_times == 0:
            print("Copying checkpoint to the buckets start", flush=True)
            self.syn_files()
            print("Copying checkpoint to the buckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.cache_dir, self.bucket), name="checkpoint_sync")
        process.start()

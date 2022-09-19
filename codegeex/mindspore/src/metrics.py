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
Eval metrics
"""

import math

from mindspore import context
from mindspore.communication.management import get_rank, get_group_size
from mindspore.nn.metrics import Metric


class PPLMetric(Metric):
    """
    Ppl metric
    """

    def __init__(self, data_length):
        super(PPLMetric, self).__init__()
        self.clear()
        self.data_length = data_length
        pipeline_stages = context.get_auto_parallel_context("pipeline_stages")
        per_stage_device_num = get_group_size() // pipeline_stages
        stage_id = get_rank() // per_stage_device_num
        self.is_last_stage = (stage_id == pipeline_stages - 1)

    def clear(self):
        """Clear the internal evaluation result."""
        self.PPL = []
        self.tokens_count = 0

    def update(self, *inputs):  # inputs
        """Update list of ppl"""
        if not self.is_last_stage:
            return
        logits = inputs[0].asnumpy().flatten().tolist()  # logits
        self.PPL.append(logits[0] * self.data_length)
        self.tokens_count += 1

    def eval(self):
        if not self.is_last_stage:
            return 0
        if self.tokens_count == 0:
            print("Warning: tokens_count is 0")
            return 0
        val_loss = sum(self.PPL) / (self.tokens_count * self.data_length)
        ppl = math.exp(min(20, val_loss))
        # print("====" * 20 + " ppl  end")
        # print("====" * 20 + " ppl: {}".format(ppl))
        # return ppl
        return val_loss


class ValidationLoss(Metric):
    def __init__(self, data_length):
        super(ValidationLoss, self).__init__()
        self.clear()
        self.data_length = data_length
        pipeline_stages = context.get_auto_parallel_context("pipeline_stages")
        per_stage_device_num = get_group_size() // pipeline_stages
        stage_id = get_rank() // per_stage_device_num
        self.is_last_stage = (stage_id == pipeline_stages - 1)

    def clear(self):
        """Clear the internal evaluation result."""
        self.metric = []
        self.tokens_count = 0

    def update(self, *inputs):  # inputs
        """Update list of ppl"""
        # logits = inputs[0].asnumpy()
        # if self.rank % 8 == 0:
        #     print("====" * 2 + " logits: {}".format(logits), flush=True)
        # self.metric.append(logits)
        if not self.is_last_stage:
            return
        logits = inputs[0].asnumpy().flatten().tolist()  # logits
        self.metric.append(logits[0] * self.data_length)
        self.tokens_count += 1

    def eval(self):
        if not self.is_last_stage == 0:
            return 0
        val_loss = sum(self.metric) / (self.tokens_count * self.data_length)
        return val_loss

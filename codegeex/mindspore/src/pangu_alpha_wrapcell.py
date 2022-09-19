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
"""GPT training wrapper"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context, Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_enable_parallel_optimizer
from src.utils import GlobalNorm, ClipByGlobalNorm

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    # 0 for clip_by_value and 1 for clip_by_norm
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad,
            F.cast(F.tuple_to_array((-clip_value,)), dt),
            F.cast(F.tuple_to_array((clip_value,)), dt),
        )
    else:
        new_grad = nn.ClipByNorm()(
            grad, F.cast(F.tuple_to_array((clip_value,)), dt)
        )
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Cast()(reciprocal(scale), F.dtype(grad))


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


class PanguAlphaTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(
            self,
            network,
            optimizer,
            scale_update_cell=None,
            enable_global_norm=False,
            config=None,
    ):
        super(PanguAlphaTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.config = config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.default_lr = Tensor([0.0], dtype=mstype.float32)
        self.enable_global_norm = enable_global_norm
        self.enable_offload = config.enable_offload
        self.clip_value = Tensor([1.0], dtype=mstype.float32)
        if config.enable_offload:
            self.clip = GlobalNorm(self.weights, config)
        else:
            self.clip = ClipByGlobalNorm(self.weights, config)
        self.cast = P.Cast()

    def construct(self, input_ids, input_position, attention_mask, layer_past=None, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        # Forward process
        loss = self.network(input_ids, input_position, attention_mask)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # Backward process using loss scale
        grads = self.grad(self.network, weights)(
            input_ids, input_position, attention_mask,
            scaling_sens_filled)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        clip_value = self.clip_value
        if self.enable_global_norm:
            grads, clip_value = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        # Check whether overflow
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # If overflow, surpass weights update
        # if not, update weights
        if not overflow:
            if self.enable_offload:
                self.optimizer(grads, clip_value)
            else:
                self.optimizer(grads)
        return loss, cond, scaling_sens


class PanguAlphaTrainPipelineWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of PanguAlpha network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, config, scale_update_cell=None, enable_global_norm=True):
        super(PanguAlphaTrainPipelineWithLossScaleCell, self).__init__(auto_prefix=False)
        self.config = config
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.reshape = P.Reshape()
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        self.clip = ClipByGlobalNorm(self.weights, self.config)
        self.micro_size = config.parallel_config.micro_batch_num
        self.opt_shard = _get_enable_parallel_optimizer()

    @C.add_flags(has_effect=True)
    def construct(
            self,
            input_ids,
            input_position,
            attention_mask,
            past=None,
            sens=None,
    ):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids, input_position, attention_mask)
        if sens is None:
            scaling_sens = self.loss_scale
            scaling_sens = self.reshape(scaling_sens, (1,))
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(
            input_ids,
            input_position,
            attention_mask,
            self.cast(scaling_sens / self.micro_size, mstype.float32),
        )
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, status_clear)
        # apply grad reducer on grads
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)
        if self.enable_global_norm:
            grads, _ = self.clip(grads)
        else:
            grads = self.hyper_map(
                F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE),
                grads)
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, overflow, scaling_sens)

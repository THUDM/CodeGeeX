# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math
import torch
from torch.nn import LayerNorm

from codegeex.megatron import get_args
from codegeex.megatron import mpu
from codegeex.megatron.model.module import MegatronModule
from codegeex.megatron.model.utils import fast_gelu

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        scale: int = 4,
    ):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            scale * args.hidden_size,
            gather_output=False,
            init_method=init_method,
            # skip_bias_add=True,
        )

        self.activation_func = fast_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            scale * args.hidden_size,
            args.hidden_size,
            input_is_parallel=True if args.tensor_model_parallel_size > 1 else False,
            init_method=output_layer_init_method,
            # skip_bias_add=True,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)

        return output, output_bias


class ParallelSelfAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        super(ParallelSelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(
            args.hidden_size // 2 if args.compress else args.hidden_size,
            world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            args.hidden_size // 2 if args.compress else args.hidden_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)
        if hasattr(args, 'attention_upweight'):
            self.attention_upweight = args.attention_upweight
        else:
            self.attention_upweight = None
        # Strided linear layer.
        self.query = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size // 2 if args.compress else args.hidden_size,
            gather_output=False,
            init_method=init_method)
        self.key = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size // 2 if args.compress else args.hidden_size,
            gather_output=False,
            init_method=init_method)
        self.value = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size // 2 if args.compress else args.hidden_size,
            gather_output=False,
            init_method=init_method)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.softmax = torch.nn.Softmax(dim=-1)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            args.hidden_size // 2 if args.compress else args.hidden_size,
            args.hidden_size,
            input_is_parallel=True if args.tensor_model_parallel_size > 1 else False,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            prompt_length=None,
            context_length=None,
    ):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        query_layer, _ = self.query(hidden_states)
        key_layer, _ = self.key(hidden_states)
        value_layer, _ = self.value(hidden_states)

        new_query_layer_shape = query_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_query_layer_shape)

        new_query_layer_shape = key_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        key_layer = key_layer.view(*new_query_layer_shape)

        new_query_layer_shape = value_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        value_layer = value_layer.view(*new_query_layer_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, sq, sk]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.contiguous().view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.contiguous().view(output_size[3], output_size[0] * output_size[1], -1)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.matmul(query_layer.transpose(0, 1),
                                     key_layer.transpose(0, 1).transpose(1, 2)) / self.norm_factor

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        if self.attention_upweight is not None and layer_past is None:
            log_attention_weights = torch.zeros(attention_scores.size(3), attention_scores.size(3),
                                                device=torch.cuda.current_device(),
                                                dtype=torch.half if self.fp16 else torch.float32)
            if prompt_length is None:
                log_attention_weights = self.attention_upweight
            else:
                log_attention_weights[:prompt_length, :prompt_length] = self.attention_upweight
            attention_scores += log_attention_weights

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if context_length is not None:
            attention_mask = torch.clone(attention_mask)
            attention_mask[:, :, context_length:, :] = True

        # attention scores and attention mask [b, np, sq, sk]
        # attention_scores = attention_mask_func(attention_scores, attention_mask)
        attention_scores = attention_scores - attention_mask * 10000.0
        if self.attention_softmax_in_fp32:
            attention_probs = self.softmax(attention_scores.float()).half()
        else:
            attention_probs = self.softmax(attention_scores.half())

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sq, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sq, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        context_layer = torch.bmm(attention_probs, value_layer.unsqueeze(0).transpose(1, 2).squeeze(0))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


class ParallelTopQuerySelfAttention(MegatronModule):
    """Parallel top query self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        super(ParallelTopQuerySelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        self.layer_number = max(1, layer_number)

        if hasattr(args, 'attention_upweight_top'):
            self.attention_upweight = args.attention_upweight_top
        else:
            self.attention_upweight = None
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(args.hidden_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            args.hidden_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        self.query = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            gather_output=False,
            init_method=init_method)

        self.key = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            gather_output=False,
            init_method=init_method)

        self.value = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.hidden_size,
            gather_output=False,
            init_method=init_method)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.softmax = torch.nn.Softmax(dim=-1)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            input_is_parallel=True if args.tensor_model_parallel_size > 1 else False,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(
            self,
            hidden_states,
            query_hidden_state,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            prompt_length=None,
            context_length=None,
    ):

        # hidden_states: [sq, b, h]

        query_layer, _ = self.query(query_hidden_state)
        key_layer, _ = self.key(hidden_states)
        value_layer, _ = self.value(hidden_states)

        new_query_layer_shape = query_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_query_layer_shape)

        new_query_layer_shape = key_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        key_layer = key_layer.view(*new_query_layer_shape)

        new_query_layer_shape = value_layer.size()[:-1] + \
                                (self.num_attention_heads_per_partition,
                                 self.hidden_size_per_attention_head)
        value_layer = value_layer.view(*new_query_layer_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, sq, sk]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [s, b, np, hn] -> [s, b * np, hn]
        query_layer = query_layer.contiguous().view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.contiguous().view(output_size[3], output_size[0] * output_size[1], -1)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.matmul(query_layer.transpose(0, 1),
                                     key_layer.transpose(0, 1).transpose(1, 2)) / self.norm_factor

        # change view to [b, np, s, s]
        attention_scores = matmul_result.view(*output_size)

        if self.attention_upweight is not None and layer_past is None:
            log_attention_weights = torch.zeros(attention_scores.size(3), attention_scores.size(3),
                                                device=torch.cuda.current_device(),
                                                dtype=torch.half if self.fp16 else torch.float32)
            if prompt_length is None:
                log_attention_weights = self.attention_upweight
            else:
                log_attention_weights[:prompt_length, :prompt_length] = self.attention_upweight
            attention_scores += log_attention_weights

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                                     ...,
                                     attention_scores.size(3) - 1,
                                     :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                                     ...,
                                     :attention_scores.size(3),
                                     :attention_scores.size(3)]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if context_length is not None:
            attention_mask = torch.clone(attention_mask)
            attention_mask[:, :, context_length:, :] = True

        # attention scores and attention mask [b, np, sq, sk]
        # attention_scores = attention_mask_func(attention_scores, attention_mask)
        attention_scores = attention_scores - attention_mask * 10000.0
        if self.attention_softmax_in_fp32:
            attention_probs = self.softmax(attention_scores.float()).half()
        else:
            attention_probs = self.softmax(attention_scores.half())

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sq, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sq, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.unsqueeze(0).transpose(1, 2).squeeze(0))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(init_method,
                                               output_layer_init_method,
                                               layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)
        if hasattr(args, 'attention_upweight'):
            self.attention_upweight = args.attention_upweight
        else:
            self.attention_upweight = None
        if hasattr(args, 'ln_fp16'):
            self.ln_fp16 = args.ln_fp16
        else:
            self.ln_fp16 = False
            # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method,
                               scale=2 if args.compress else 4)

    def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            prompt_length=None,
            context_length=None,
    ):
        # hidden_states: [b, s, h]
        if self.ln_fp16:
            layernorm_output = self.input_layernorm(hidden_states)
        else:
            layernorm_output = self.input_layernorm(hidden_states.float()).half()

        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value,
                           prompt_length=prompt_length,
                           context_length=context_length)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        if self.ln_fp16:
            layernorm_output = self.post_attention_layernorm(layernorm_input)
        else:
            layernorm_output = self.post_attention_layernorm(layernorm_input.float()).half()

        mlp_output, _ = self.mlp(layernorm_output)

        # MLP.
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, "_args"):
                self._args = get_args()
            hidden_states, attention_mask = inputs, self._args.attn_mask
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError("Received more inputs than understood.")


class ParallelTopQueryLayer(MegatronModule):
    """A single top query layer.

    Top query layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTopQueryLayer, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelTopQuerySelfAttention(init_method,
                                                       output_layer_init_method,
                                                       layer_number)

        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if hasattr(args, 'ln_fp16'):
            self.ln_fp16 = args.ln_fp16
        else:
            self.ln_fp16 = False

        # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)

    def forward(
            self,
            hidden_states,
            query_hidden_state,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            prompt_length=None,
            context_length=None,
    ):
        # hidden_states: [b, s, h]
        assert query_hidden_state != None

        # Layer norm at the beginning of the transformer layer.
        if self.ln_fp16:
            layernorm_output = self.input_layernorm(hidden_states)
        else:
            layernorm_output = self.input_layernorm(hidden_states.float()).half()

        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           query_hidden_state,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value,
                           prompt_length=prompt_length,
                           context_length=context_length)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        if self.ln_fp16:
            layernorm_output = self.post_attention_layernorm(layernorm_input)
        else:
            layernorm_output = self.post_attention_layernorm(layernorm_input.float()).half()

        # MLP.
        mlp_output, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTopQueryLayerPipe(ParallelTopQueryLayer):
    """Extends ParallelTopQueryLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 2:
            # No attention mask forwarded, search for args.attn_mask
            if not hasattr(self, "_args"):
                self._args = get_args()
            hidden_states, query_hidden_state = inputs
            attention_mask = self._args.attn_mask
            return super().forward(hidden_states, query_hidden_state, attention_mask, **kwargs)
        elif len(inputs) == 3:
            # Attention mask is an activation.
            hidden_states, query_hidden_state, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        else:
            raise RuntimeError("Received more inputs than understood.")
        
        
class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = None

        #################
        assert self.num_unique_layers is None
        #################

        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = 'grouped'

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method, layer_number)

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_unique_layers)])

        self.topQueryLayer = ParallelTopQueryLayer(
            init_method,
            output_layer_init_method, self.num_unique_layers)

        # Final layer norm before output.
        if hasattr(args, 'ln_fp16'):
            self.ln_fp16 = args.ln_fp16
        else:
            self.ln_fp16 = False

        self.final_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers)
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
        
    def forward(
            self,
            hidden_states,
            query_hidden_state,
            attention_mask,
            layer_past=None,
            get_key_value=False,
            prompt_length=None,
            context_length=None,
    ):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        query_hidden_state = query_hidden_state.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value,
                                      prompt_length=prompt_length,
                                      context_length=context_length)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        if self.ln_fp16:
            hidden_states_ = self.final_layernorm(hidden_states)
        else:
            hidden_states_ = self.final_layernorm(hidden_states.float()).half()

        #################################
        # top query layer
        #################################
        past = None
        if layer_past is not None:
            past = layer_past[self.num_layers]
        hidden_states = self.topQueryLayer(hidden_states_,
                                           query_hidden_state,
                                           attention_mask,
                                           layer_past=past,
                                           get_key_value=get_key_value,
                                           prompt_length=prompt_length,
                                           context_length=context_length)

        if get_key_value:
            hidden_states, present = hidden_states
            presents.append(present)

        # reverting data format change [s b h] --> [b s h]
        output = hidden_states.transpose(0, 1).contiguous()

        if get_key_value:
            output = [output, presents]

        return output

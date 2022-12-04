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

import torch
from codegeex.megatron import get_args, mpu
from codegeex.megatron.model import LayerNorm
from codegeex.megatron.enums import AttnMaskType
from codegeex.megatron.model.module import MegatronModule
from codegeex.megatron.model.language_model import parallel_lm_logits, get_language_model, EmbeddingPipe, QueryEmbeddingPipe
from codegeex.megatron.model.utils import init_method_normal, scaled_init_method_normal
from codegeex.megatron.model.transformer import ParallelTransformerLayerPipe, ParallelTopQueryLayerPipe
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec


class CodeGeeXModel(MegatronModule):
    """Code Generation Model for Multilingual Program Synthesis."""

    def __init__(self, num_tokentypes=0, parallel_output=False):
        super(CodeGeeXModel, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)
        
    def forward(
            self,
            input_ids,
            position_ids,
            attention_mask,
            labels=None,
            tokentype_ids=None,
            layer_past=None,
            get_key_value=False,
            forward_method_parallel_output=None,
            prompt_length=None,
            context_length=None,
    ):

        # Language model.
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value,
                                        prompt_length=prompt_length,
                                        context_length=context_length)

        if get_key_value:
            lm_output, presents = lm_output

        lm_output = torch.add(lm_output, 0)
        # Output.
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output)

        if get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = mpu.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)

            return loss

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)


def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class CodeGeeXModelPipe(PipelineModule, MegatronModule):
    """Pipeline version of CodeGeeX."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)

        self.specs = []
        
        # Embedding layer
        self.specs.append(
            TiedLayerSpec(
                "embed",
                EmbeddingPipe,
                args.hidden_size,
                args.padded_vocab_size,
                args.max_position_embeddings,
                args.hidden_dropout,
                init_method=init_method,
                num_tokentypes=num_tokentypes,
                tied_weight_attr="word_embeddings_weight",
            )
        )

        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(
                    ParallelTransformerLayerPipe,
                    init_method=init_method,
                    output_layer_init_method=scaled_init_method_normal(
                        args.init_method_std, args.num_layers
                    ),
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal,
                )
            )

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        # Final layernorm after transformer layers
        self.specs.append(
            LayerSpec(LayerNorm, args.hidden_size, eps=args.layernorm_epsilon)
        )

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline."""
            return parallel_lm_logits(
                lm_output, embedding.word_embeddings_weight, self.parallel_output
            )

        self.specs.append(
            TiedLayerSpec(
                "embed",
                EmbeddingPipe,
                args.hidden_size,
                args.padded_vocab_size,
                args.max_position_embeddings,
                args.hidden_dropout,
                init_method=init_method,
                num_tokentypes=num_tokentypes,
                forward_fn=_logits_helper,
                tied_weight_attr="word_embeddings_weight",
            )
        )

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

        topo = PipeModelDataParallelTopology(
            num_pp=mpu.get_pipeline_model_parallel_world_size(),
            num_mp=mpu.get_tensor_model_parallel_world_size(),
            num_dp=mpu.get_data_parallel_world_size(),
        )

        super().__init__(
            layers=self.specs,
            loss_fn=CrossEntropy,
            topology=topo,
            activation_checkpoint_interval=interval,
            partition_method="type:transformer",
        )

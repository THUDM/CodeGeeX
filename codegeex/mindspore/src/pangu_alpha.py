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
"""PanguAlpha model"""
import copy
import os

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
# from mindspore.parallel.nn.layers import _LayerNorm
from mindspore.nn.transformer.layers import _Dropout, _LayerNorm
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel.nn import MoEConfig
from mindspore.parallel.nn.transformer import (
    VocabEmbedding,
    TransformerEncoder,
    TransformerEncoderLayer,
    AttentionMask,
)


class EmbeddingLayer(nn.Cell):
    r"""Embedding layer of the PanGUAlpha Model"""

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        # Only for the pipeline mode, the embedding needs to be row sliced.
        dp = config.parallel_config.embedding_dp_mp_config.data_parallel
        mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        self.word_embedding = VocabEmbedding(
            vocab_size=(config.vocab_size // 1024 + 1) * 1024,
            embedding_size=config.hidden_size,
            param_init=initializer(
                "normal",
                [(config.vocab_size // 1024 + 1) * 1024, config.hidden_size],
                # dtype=config.param_init_type,
                dtype=mstype.float32,
            ),
            parallel_config=config.parallel_config.embedding_dp_mp_config,
        )
        self.word_embedding.gather.shard(((mp, 1), (dp, 1)))
        # self.word_embedding.embedding_table.parallel_optimizer = True
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.position_embedding = VocabEmbedding(
            vocab_size=config.seq_length,
            embedding_size=config.hidden_size,
            param_init=initializer(
                "normal",
                [config.seq_length, config.hidden_size],
                # dtype=config.param_init_type,
                dtype=mstype.float32,
            ),
            parallel_config=copied_parallel_config.embedding_dp_mp_config,
        )
        self.split = P.Split(1, 2).shard(
            ((config.parallel_config.data_parallel, 1, 1),)
        )
        self.print = P.Print()
        self.add = P.Add().shard(
            (
                (config.parallel_config.data_parallel, 1, 1),
                (config.parallel_config.data_parallel, 1, 1),
            )
        )
        # self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout = _Dropout(1 - config.dropout_rate)
        self.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))
        # self.dropout.dropout.shard(
        #    ((config.parallel_config.data_parallel, 1, 1),)
        # )
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size

    def construct(
            self, input_ids, input_position, init_reset, batch_valid_length
    ):
        word_embedding, word_table = self.word_embedding(input_ids)
        if self.use_past and not self.is_first_iteration:
            _, seq_length = F.shape(input_ids)
            # self.print("==batch_valid_length is: ", batch_valid_length, ", transform into: ", self.batch_size, "*", seq_length)
            input_position = batch_valid_length.view(self.batch_size, seq_length)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table

    def get_word_embedding_weight(self):
        return self.word_embedding.embedding_table


class QueryLayer(TransformerEncoderLayer):
    r"""Query Layer at the final layer."""

    def __init__(
            self,
            batch_size,
            hidden_size,
            ffn_hidden_size,
            num_heads,
            seq_length,
            attention_dropout_rate=0.1,
            hidden_dropout_rate=0.1,
            post_layernorm_residual=False,
            param_init_type=mstype.float32,
            hidden_act="fast_gelu",
            use_past=False,
            parallel_config=None,
            softmax_compute_type=mstype.float32,
    ):
        super(QueryLayer, self).__init__(
            batch_size=batch_size,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_heads=num_heads,
            seq_length=seq_length,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            post_layernorm_residual=post_layernorm_residual,
            param_init_type=param_init_type,
            hidden_act=hidden_act,
            use_past=use_past,
            parallel_config=parallel_config.dp_mp_config,
            softmax_compute_type=softmax_compute_type,
        )

    def construct(
            self,
            x,
            query_vector,
            input_mask,
            init_reset=True,
            batch_valid_length=None,
    ):
        r"""
        The forward process of the block.
        """
        # [bs * seq_length, embedding_size]
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(
                self.key_past,
                self.mul(self.key_past, F.cast(init_reset, self.dtype)),
            )
            value_reset = self.assign(
                self.value_past,
                self.mul(self.value_past, F.cast(init_reset, self.dtype)),
            )
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(
            query_vector,
            input_x,
            input_x,
            input_mask,
            self.key_past,
            self.value_past,
            batch_valid_length,
        )
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
        return output, layer_present


class PanGuHead(Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(
            self, hidden_size, compute_type=mstype.float16, parallel_config=None
    ):
        super(PanGuHead, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(
                ((parallel_config.data_parallel, 1), (1, 1))
            )
        else:
            print("====vocab_emb_dp is false", flush=True)
            self.matmul = P.MatMul(transpose_b=True).shard(
                (
                    (parallel_config.data_parallel, 1),
                    (parallel_config.model_parallel, 1),
                )
            )
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(
            self.cast(state, self.dtype), self.cast(embed, self.dtype)
        )
        return logits


def set_parallel_configure_for_layer(
        network, layer_id, offset, parallel_config, layers
):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id
    print(f"pipeline stage id is {pp_id}", flush=True)

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if parallel_config.recompute:
        network.recompute(recompute_slice_activation=True)


class PanguAlpha_Model(Cell):
    r"""The base backbone of the PanGuAlpha model"""

    def __init__(self, config):
        super(PanguAlpha_Model, self).__init__()
        self.is_pipeline = config.parallel_config.pipeline_stage > 1
        self.embedding = EmbeddingLayer(config)
        self.config = config
        self.layernorm = _LayerNorm((config.hidden_size,)).to_float(
            mstype.float32
        )
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(
                config.parallel_config.gradient_aggregation_group
            )
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = (
                config.parallel_config.pipeline_stage - 1
        )
        # Configure the shard configure of the Embedding layer
        self.embedding.pipeline_stage = 0
        self.num_layers = config.num_layers
        if config.use_moe:
            moe_config = MoEConfig(
                expert_num=config.parallel_config.data_parallel
                           * config.per_dp_dim_expert_num
            )
        else:
            moe_config = MoEConfig(expert_num=1)
        # The shard setting of Transformer is set within the class StackedTransformer
        self.blocks = TransformerEncoder(num_layers=config.num_layers - 1,
                                         batch_size=config.batch_size,
                                         hidden_size=config.hidden_size,
                                         ffn_hidden_size=config.ffn_hidden_size,
                                         num_heads=config.num_heads,
                                         seq_length=config.seq_length,
                                         attention_dropout_rate=config.dropout_rate,
                                         hidden_dropout_rate=config.dropout_rate,
                                         lambda_func=set_parallel_configure_for_layer,
                                         hidden_act="fast_gelu",
                                         param_init_type=config.param_init_type,
                                         use_past=config.use_past,
                                         parallel_config=config.parallel_config,
                                         moe_config=moe_config,
                                         softmax_compute_type=config.softmax_compute_type).blocks
        for block in self.blocks:
            block.attention.dense1.bias.parallel_optimizer = False
            block.attention.dense2.bias.parallel_optimizer = False
            block.attention.dense3.bias.parallel_optimizer = False
            block.output.mapping.bias.parallel_optimizer = False
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        self.top_query_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                  embedding_size=config.hidden_size,
                                                  param_init=initializer("normal",
                                                                         [config.seq_length, config.hidden_size],
                                                                         dtype=mstype.float32),
                                                  # dtype=config.param_init_type),
                                                  parallel_config=copied_parallel_config.embedding_dp_mp_config)
        self.top_query_embedding.pipeline_stage = config.parallel_config.pipeline_stage - 1
        if config.parallel_config.pipeline_stage > 1:
            self.top_query_embedding.set_comm_fusion(2)
        else:
            self.top_query_embedding.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

        self.top_query_layer = QueryLayer(batch_size=config.batch_size,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.ffn_hidden_size,
                                          num_heads=config.num_heads,
                                          seq_length=config.seq_length,
                                          attention_dropout_rate=config.dropout_rate,
                                          hidden_dropout_rate=config.dropout_rate,
                                          hidden_act=config.hidden_act,
                                          param_init_type=config.param_init_type,
                                          use_past=config.use_past,
                                          parallel_config=config.parallel_config)
        self.top_query_layer.attention.dense1.bias.parallel_optimizer = False
        self.top_query_layer.attention.dense2.bias.parallel_optimizer = False
        self.top_query_layer.attention.dense3.bias.parallel_optimizer = False
        self.top_query_layer.output.mapping.bias.parallel_optimizer = False
        if config.parallel_config.recompute:
            self.top_query_layer.recompute()
        self.top_query_layer.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.top_query_layer.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.dtype = mstype.float16

        # if config.load_ckpt_path:
        #     self.load_embedding_from_ckpt(config.load_ckpt_path)

    def construct(self, input_ids,
                  input_position,
                  encoder_masks,
                  init_reset=True,
                  batch_valid_length=None):
        r"""forward pass of the model"""
        embed, word_table = self.embedding(input_ids, input_position, init_reset, batch_valid_length)
        hidden_state = P.Cast()(embed, self.dtype)
        if init_reset is False:
            hidden_state = self.reshape_to_2d(hidden_state)
        # encoder_mask = self.create_encoder_mask(encoder_masks)
        if self.blocks is not None:
            for i in range(self.num_layers - 1):
                hidden_state, _ = self.blocks[i](hidden_state, encoder_masks, init_reset, batch_valid_length)
        if self.is_pipeline:
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(hidden_state, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)
            encoder_output = self.layernorm(encoder_output)
        else:
            hidden_state = self.reshape_to_2d(hidden_state)
            encoder_output = self.layernorm(hidden_state)
            encoder_output = P.Cast()(encoder_output, self.dtype)
            top_query_hidden_states, _ = self.top_query_embedding(input_position)
            top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
            encoder_output, _ = self.top_query_layer(encoder_output, top_query_hidden_states,
                                                     encoder_masks, init_reset, batch_valid_length)

        return encoder_output, word_table

    def reshape_to_2d(self, x):
        r"""reshape nd tensor to 2d, if n <= 2, keep original shape."""
        shape = F.shape(x)
        if len(shape) <= 2:
            return x
        x = F.reshape(x, (-1, shape[-1]))
        return x

    def load_embedding_from_ckpt(self, load_ckpt_path):
        r"""load the weights from the checkpoint"""

        def load_param(path):
            if os.path.exists(path):
                p_table = np.load(path)
                table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(
                    f"{path} file not exits, "
                    f"please check whether embedding file exit."
                )
            return table_param

        # three embedding needed to be loaded
        # Loading the embedding table from the ckpt path:
        position_embedding_path = os.path.join(load_ckpt_path, 'position_embedding.npy')
        word_embedding_path = os.path.join(load_ckpt_path, 'word_embedding.npy')
        top_query_embedding_path = os.path.join(load_ckpt_path, 'top_query_embedding.npy')
        self.embedding.word_embedding.embedding_table = Parameter(initializer(load_param(word_embedding_path),
                                                                              [self.config.vocab_size,
                                                                               self.config.hidden_size]),
                                                                  name='word_embedding_table', parallel_optimizer=False)
        self.embedding.position_embedding.embedding_table = Parameter(initializer(load_param(position_embedding_path),
                                                                                  [self.config.seq_length,
                                                                                   self.config.hidden_size]),
                                                                      name='position_embedding_table',
                                                                      parallel_optimizer=False)
        self.top_query_embedding.embedding_table = Parameter(initializer(load_param(top_query_embedding_path),
                                                                         [self.config.seq_length,
                                                                          self.config.hidden_size]),
                                                             name='query_embedding_table', parallel_optimizer=False)


class PanguAlphaModel(nn.Cell):
    """
    The PanguAlpha network consisting of two parts the backbone and the head
    Args:
        config(PanguAlphaConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PanguAlphaModel, self).__init__()
        # Network head to get logits over vocabulary
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        self.head = PanGuHead(
            hidden_size=config.hidden_size,
            parallel_config=copied_parallel_config,
        )
        self.head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.backbone = PanguAlpha_Model(config)
        self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

    def construct(self, input_ids, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        output_states, word_table = self.backbone(input_ids, input_position, attention_mask,
                                                  init_reset, batch_valid_length)
        logits = self.head(output_states, word_table)
        return logits


class PanGUAlphaWithLoss(Cell):
    """
    PanguAlpha training loss for generation.
    Args:
        config(PanGUConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss):
        super(PanGUAlphaWithLoss, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.eod_token = config.eod_token
        self.loss = loss

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        # self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        self.print = P.Print()
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num

    def construct(self, input_ids, input_position=None, attention_mask=None):
        r"""Forward process of the pangu alpha model"""
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        # P.Print()("==net tokens is:", tokens)
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        decoder_attention_masks = self.slice2(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                              (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eod_token), mstype.float32)
        logits = self.network(tokens, input_position, decoder_attention_masks)
        # P.Print()("==logits_is:", logits, ",shape is:", logits.shape)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        # P.Print()("==input_mask is:", input_mask)
        output = self.loss(logits, labels, input_mask)
        # P.Print()("==net output is:", output)
        return output


class EvalNet(nn.Cell):
    """
    PanguAlpha evaluation net
    Args:
        backbone: backbone network of PanguAlpha
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=6, seq_length=2048):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)
        self.not_equal = P.NotEqual().shard(((1, 1), ()))

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        # input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        input_mask = F.cast(self.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        if self.is_first_iteration is False:
            attention_mask = P.Tile()(self.all_ones_attention_mask, (bs, 1, 1))
        else:
            attention_mask = self.get_attention_mask(input_mask)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)
        index = current_index.view(-1, )
        # P.Print()("==logits_is:", logits, ",shape is:", logits.shape)
        # P.Print()("==index_is:", index, ",shape is:", index.shape)
        logits = self.gather(logits, index, 0)
        logits = logits.view(bs, 1, -1)
        log_probs = self.log_softmax(logits)
        return log_probs


class LogitsNet(nn.Cell):
    """
    PanguAlpha evaluation net
    Args:
        backbone: backbone network of PanguAlpha
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=6, seq_length=2048):
        super(LogitsNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.Gather().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)
        self.not_equal = P.NotEqual().shard(((1, 1), ()))

    def construct(self, input_ids, init_reset=True, batch_valid_length=None, attention_mask=None):
        """evaluation net"""
        # input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        input_mask = F.cast(self.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        if attention_mask is None:
            if self.is_first_iteration is False:
                attention_mask = P.Tile()(self.all_ones_attention_mask, (bs, 1, 1))
            else:
                attention_mask = self.get_attention_mask(input_mask)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)

        return logits


class PanGUAlphaWithFinetuneLoss(Cell):
    """
    PanguAlpha training loss for generation.
    Args:
        config(PanGUConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss):
        super(PanGUAlphaWithFinetuneLoss, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.eod_token = config.eod_token
        self.loss = loss

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        # self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        self.print = P.Print()
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num

    def construct(self, input_ids, loss_mask, input_position, attention_mask):
        r"""Forward process of the pangu alpha model"""
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        # P.Print()("==net tokens is:", tokens)
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        decoder_attention_masks = self.slice2(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                              (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eod_token), mstype.float32)
        logits = self.network(tokens, input_position, decoder_attention_masks)
        # P.Print()("===logits: ", logits, ", shape: ", logits.shape)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(loss_mask, (-1,))
        # P.Print()("===labels: ", labels, ", shape: ", labels.shape)
        # input_mask = P.Reshape()(input_mask, (-1,))
        # P.Print()("===input_mask: ", input_mask, ", shape: ", input_mask.shape)
        output = self.loss(logits, labels, input_mask)
        # P.Print()("==net output is:", output)
        return output

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

"""Merge model parallel partitions."""

import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from codegeex.megatron import get_args
from codegeex.megatron.model import CodeGeeXModel
from codegeex.megatron.initialize import initialize_megatron
from codegeex.megatron.checkpointing import ensure_directory_exists


def get_change_ckpt_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='Mindspore to megatron')
    group.add_argument(
        '--npy-ckpt-path',
        type=str,
        required=True,
        help='path of npy checkpoint.',
    )
    group.add_argument(
        '--save-ckpt-path',
        type=str,
        required=True,
        help='path to save checkpoint.',
    )

    return parser


def loadModelFromNp(sd, args):
    num_layers = args.num_layers
    npCkptPath = args.npy_ckpt_path
    languageModel = sd['module']['language_model']
    loadEmbeddingFromNp(npCkptPath, languageModel)
    transformer = sd['module']['language_model']['transformer']
    for layerID in range(num_layers):
        loadAttentionLayerFromNp(npCkptPath, transformer, layerID)
    loadQueryLayerFromNp(npCkptPath, transformer)

    transformer['final_layernorm.weight'][:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.layernorm.gamma.npy')
        ).float()
    transformer['final_layernorm.bias'][:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.layernorm.beta.npy')
        ).float()


def loadEmbeddingFromNp(npCkptPath, languageModel, vocabSize=52224):
    word_embedding_np = \
        np.load(npCkptPath + 'backbone.embedding.word_embedding.embedding_table.npy')
    languageModel['embedding']['word_embeddings']['weight'][:vocabSize, :] = \
        torch.tensor(word_embedding_np).float()

    position_embeddings_np = \
        np.load(npCkptPath + 'backbone.embedding.position_embedding.embedding_table.npy')
    languageModel['embedding']['position_embeddings']['weight'][:, :] = \
        torch.tensor(position_embeddings_np).float()

    topQueryEmbedding_np = \
        np.load(npCkptPath + 'backbone.top_query_embedding.embedding_table.npy')
    languageModel['topQueryEmbedding']['top_query_embeddings']['weight'][:, :] = \
        torch.tensor(topQueryEmbedding_np).float()


def loadAttentionLayerFromNp(npCkptPath, transformer, layerID):
    attention_dense1_weight_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense1.weight.npy')
    attention_dense2_weight_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense2.weight.npy')
    attention_dense3_weight_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense3.weight.npy')

    attention_dense1_bias_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense1.bias.npy')
    attention_dense2_bias_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense2.bias.npy')
    attention_dense3_bias_np = \
        np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.dense3.bias.npy')

    query_weight = transformer[f'layers.{layerID}.attention.query.weight']
    key_weight = transformer[f'layers.{layerID}.attention.key.weight']
    value_weight = transformer[f'layers.{layerID}.attention.value.weight']

    query_weight[:] = torch.tensor(attention_dense1_weight_np).float()
    key_weight[:] = torch.tensor(attention_dense2_weight_np).float()
    value_weight[:] = torch.tensor(attention_dense3_weight_np).float()

    query_bias = transformer[f'layers.{layerID}.attention.query.bias']
    key_bias = transformer[f'layers.{layerID}.attention.key.bias']
    value_bias = transformer[f'layers.{layerID}.attention.value.bias']

    query_bias[:] = torch.tensor(attention_dense1_bias_np).float()
    key_bias[:] = torch.tensor(attention_dense2_bias_np).float()
    value_bias[:] = torch.tensor(attention_dense3_bias_np).float()

    att_dense_weight = transformer[f'layers.{layerID}.attention.dense.weight']
    att_dense_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.projection.weight.npy').transpose()
        ).float()
    att_dense_bias = transformer[f'layers.{layerID}.attention.dense.bias']
    att_dense_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.attention.projection.bias.npy')
        ).float()

    mlp_dense_h_to_4h_weight = transformer[f'layers.{layerID}.mlp.dense_h_to_4h.weight']
    mlp_dense_h_to_4h_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.output.mapping.weight.npy').transpose()
        ).float()
    mlp_dense_h_to_4h_bias = transformer[f'layers.{layerID}.mlp.dense_h_to_4h.bias']
    mlp_dense_h_to_4h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.output.mapping.bias.npy')
        ).float()

    mlp_dense_4h_to_h_weight = transformer[f'layers.{layerID}.mlp.dense_4h_to_h.weight']
    mlp_dense_4h_to_h_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.output.projection.weight.npy').transpose()
        ).float()
    mlp_dense_4h_to_h_bias = transformer[f'layers.{layerID}.mlp.dense_4h_to_h.bias']
    mlp_dense_4h_to_h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.output.projection.bias.npy')
        ).float()

    input_layernorm_weight = transformer[f'layers.{layerID}.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'layers.{layerID}.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'layers.{layerID}.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'layers.{layerID}.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm2.beta.npy')
        ).float()

    input_layernorm_weight = transformer[f'layers.{layerID}.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'layers.{layerID}.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'layers.{layerID}.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'layers.{layerID}.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.blocks.{layerID}.layernorm2.beta.npy')
        ).float()


def loadQueryLayerFromNp(npCkptPath, transformer):
    attention_dense1_weight_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense1.weight.npy')
    attention_dense1_bias_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense1.bias.npy')
    attention_dense2_weight_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense2.weight.npy')
    attention_dense2_bias_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense2.bias.npy')
    attention_dense3_weight_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense3.weight.npy')
    attention_dense3_bias_np = \
        np.load(npCkptPath + f'backbone.top_query_layer.attention.dense3.bias.npy')

    query_weight = transformer[f'topQueryLayer.attention.query.weight']
    query_weight[:, :] = \
        torch.tensor(attention_dense1_weight_np).float()
    query_bias = transformer[f'topQueryLayer.attention.query.bias']
    query_bias[:] = torch.tensor(attention_dense1_bias_np).float()

    key_weight = transformer[f'topQueryLayer.attention.key.weight']
    key_weight[:, :] = \
        torch.tensor(attention_dense2_weight_np).float()
    key_bias = transformer[f'topQueryLayer.attention.key.bias']
    key_bias[:] = torch.tensor(attention_dense2_bias_np).float()

    value_weight = transformer[f'topQueryLayer.attention.value.weight']
    value_weight[:, :] = \
        torch.tensor(attention_dense3_weight_np).float()
    value_bias = transformer[f'topQueryLayer.attention.value.bias']
    value_bias[:] = torch.tensor(attention_dense3_bias_np).float()

    att_dense_weight = transformer[f'topQueryLayer.attention.dense.weight']
    att_dense_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.attention.projection.weight.npy')
            .transpose()
        ).float()
    att_dense_bias = transformer[f'topQueryLayer.attention.dense.bias']
    att_dense_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.attention.projection.bias.npy')
        ).float()

    mlp_dense_h_to_4h_weight = transformer[f'topQueryLayer.mlp.dense_h_to_4h.weight']
    mlp_dense_h_to_4h_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.output.mapping.weight.npy')
            .transpose()
        ).float()
    mlp_dense_h_to_4h_bias = transformer[f'topQueryLayer.mlp.dense_h_to_4h.bias']
    mlp_dense_h_to_4h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.output.mapping.bias.npy')
        ).float()

    mlp_dense_4h_to_h_weight = transformer[f'topQueryLayer.mlp.dense_4h_to_h.weight']
    mlp_dense_4h_to_h_weight[:, :] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.output.projection.weight.npy')
            .transpose()
        ).float()
    mlp_dense_4h_to_h_bias = transformer[f'topQueryLayer.mlp.dense_4h_to_h.bias']
    mlp_dense_4h_to_h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.output.projection.bias.npy')
        ).float()

    input_layernorm_weight = transformer[f'topQueryLayer.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'topQueryLayer.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'topQueryLayer.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'topQueryLayer.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath + f'backbone.top_query_layer.layernorm2.beta.npy')
        ).float()


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))

    initialize_megatron(
        extra_args_provider=get_change_ckpt_args,
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng"   : True,
            "no_load_optim" : True,
        },
    )

    args = get_args()
    model = CodeGeeXModel()
    # print(dir(model))
    print(model.state_dict)

    # Save the model.
    sd = {}
    sd['module'] = model.state_dict_for_save_checkpoint()
    ensure_directory_exists(args.save_ckpt_path)
    loadModelFromNp(sd, args)
    print('> saving merged model to {}'.format(args.save_ckpt_path))
    torch.save(sd, args.save_ckpt_path)
    print(f"Converted checkpoint saved in {args.save_ckpt_path}.")


if __name__ == '__main__':
    main()

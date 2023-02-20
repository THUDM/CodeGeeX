"""Merge model parallel partitions into a single checkpoint."""

import os
import torch
import random

from codegeex.megatron import get_args
from codegeex.megatron.model import CodeGeeXModel
from codegeex.megatron.initialize import initialize_megatron
from codegeex.megatron.checkpointing import ensure_directory_exists


def get_change_ckpt_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='Mindspore to megatron')
    group.add_argument(
        '--load-ckpt-path',
        type=str,
        required=True,
        help='dir to load model parallel partitions.',
    )
    group.add_argument(
        '--save-ckpt-path',
        type=str,
        required=True,
        help='path to save ".pt" checkpoint.',
    )
    group.add_argument(
        '--save-name',
        type=str,
        help='name of checkpoint.',
    )
    group.add_argument(
        '--source-tensor-model-parallel-size',
        type=int,
        default=2,
        help='original tensor model parallel size',
    )
    
    return parser


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
    print(model.state_dict)

    # Save the model.
    sd = {}
    sd['module'] = model.state_dict_for_save_checkpoint()
    ensure_directory_exists(args.save_ckpt_path)
    
    print(f"Load ckpt from {args.load_ckpt_path}...")
    state_dict_list = []
    for i in range(args.source_tensor_model_parallel_size):
        try:
            state_dict_list.append(torch.load(os.path.join(args.load_ckpt_path, f"mp_rank_{i:02d}_model_states.pt"), map_location="cpu"))
        except Exception as e:
            print(e)
            exit(0)
    
    print(f"Merging {len(state_dict_list)} partitions into a single ckpt...")
    print("Merging Embedding layers...")
    vocab_parallel_size = args.make_vocab_size_divisible_by // args.source_tensor_model_parallel_size
    for i in range(args.source_tensor_model_parallel_size):
        sd['module']['language_model']['embedding']['word_embeddings']['weight'][i * vocab_parallel_size : (i + 1) * vocab_parallel_size, :] = state_dict_list[i]['module']['language_model']['embedding']['word_embeddings']['weight']
    
    sd['module']['language_model']['embedding']['position_embeddings']['weight'] = state_dict_list[0]['module']['language_model']['embedding']['position_embeddings']['weight']
        
    print("Merging QueryEmbedding layers...")
    query_parallel_size = args.max_position_embeddings // args.source_tensor_model_parallel_size
    for i in range(args.source_tensor_model_parallel_size):
        sd['module']['language_model']['topQueryEmbedding']['top_query_embeddings']['weight'][i * query_parallel_size : (i + 1) * query_parallel_size, :] = state_dict_list[i]['module']['language_model']['topQueryEmbedding']['top_query_embeddings'].pop('weight', None)
    
    print("Merging Transformer layers...")
    for layer_name in sd['module']['language_model']['transformer'].keys():
        if "layernorm" in layer_name:
            sd['module']['language_model']['transformer'][layer_name] = state_dict_list[0]['module']['language_model']['transformer'].pop(layer_name, None)
        elif "attention" in layer_name and "weight" in layer_name:
            if "dense" in layer_name:
                hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[1] // args.source_tensor_model_parallel_size
                for i in range(args.source_tensor_model_parallel_size):
                    sd['module']['language_model']['transformer'][layer_name][:, i * hidden_parallel_size : (i + 1) * hidden_parallel_size] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
            else:
                hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[0] // args.source_tensor_model_parallel_size
                for i in range(args.source_tensor_model_parallel_size):
                    sd['module']['language_model']['transformer'][layer_name][i * hidden_parallel_size : (i + 1) * hidden_parallel_size, :] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
        elif "weight" in layer_name and "dense" in layer_name:
            if "h_to_4h" in layer_name:
                hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[0] // args.source_tensor_model_parallel_size
                for i in range(args.source_tensor_model_parallel_size):
                    sd['module']['language_model']['transformer'][layer_name][i * hidden_parallel_size : (i + 1) * hidden_parallel_size, :] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
            else:
                hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[1] // args.source_tensor_model_parallel_size
                for i in range(args.source_tensor_model_parallel_size):
                    sd['module']['language_model']['transformer'][layer_name][:, i * hidden_parallel_size : (i + 1) * hidden_parallel_size] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
        elif "bias" in layer_name:
            if "mlp" in layer_name:
                if "4h_to_h" in layer_name:
                    sd['module']['language_model']['transformer'][layer_name] = state_dict_list[0]['module']['language_model']['transformer'].pop(layer_name, None)
                else:
                    hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[0] // args.source_tensor_model_parallel_size
                    for i in range(args.source_tensor_model_parallel_size):
                        sd['module']['language_model']['transformer'][layer_name][i * hidden_parallel_size : (i + 1) * hidden_parallel_size] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
            elif "attention" in layer_name:
                if "dense" in layer_name:
                    sd['module']['language_model']['transformer'][layer_name] = state_dict_list[0]['module']['language_model']['transformer'].pop(layer_name, None)
                else:
                    hidden_parallel_size = sd['module']['language_model']['transformer'][layer_name].shape[0] // args.source_tensor_model_parallel_size
                    for i in range(args.source_tensor_model_parallel_size):
                        sd['module']['language_model']['transformer'][layer_name][i * hidden_parallel_size : (i + 1) * hidden_parallel_size] = state_dict_list[i]['module']['language_model']['transformer'].pop(layer_name, None)
        else:
            sd['module']['language_model']['transformer'][layer_name] = state_dict_list[0]['module']['language_model']['transformer'].pop(layer_name, None)
            
    if args.save_ckpt_path.endswith(".pt"):
        save_ckpt_path = args.save_ckpt_path
    else:
        os.makedirs(args.save_ckpt_path, exist_ok=True)
        if args.save_name:
            save_ckpt_path = os.path.join(args.save_ckpt_path, args.save_name)
        else:
            save_ckpt_path = os.path.join(args.save_ckpt_path, "mp_rank_00_model_states.pt")
    
    torch.save(sd, save_ckpt_path)
    print(f"Converted checkpoint saved in {save_ckpt_path}.")


if __name__ == '__main__':
    main()

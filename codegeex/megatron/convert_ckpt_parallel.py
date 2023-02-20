"""Get model parallel partitions."""

import os
import torch
import argparse


def get_change_ckpt_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='Mindspore to megatron')
    group.add_argument(
        '--load-ckpt-path',
        type=str,
        required=True,
        help='path to load ".pt" checkpoint.',
    )
    group.add_argument(
        '--save-ckpt-path',
        type=str,
        required=True,
        help='dir to save converted checkpoints.',
    )
    group.add_argument(
        '--target-tensor-model-parallel-size',
        type=int,
        default=2,
        help='target tensor model parallel size',
    )
    
    return parser


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser = get_change_ckpt_args(parser)
    args, _ = parser.parse_known_args()
    
    print(f"Load ckpt from {args.load_ckpt_path}...")
    state_dict = torch.load(args.load_ckpt_path, map_location="cpu")

    print(f"Spliting ckpt into {args.target_tensor_model_parallel_size} parts...")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})
    
    print("Converting Embedding layers...")
    word_embeddings = state_dict['module']['language_model']['embedding']['word_embeddings']['weight']
    position_embeddings = state_dict['module']['language_model']['embedding']['position_embeddings']['weight']
    out_word_embeddings = torch.chunk(word_embeddings, args.target_tensor_model_parallel_size, dim=0)
    
    for i in range(args.target_tensor_model_parallel_size):
        pos_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.embedding.position_embeddings"
        )
        pos_emb_dict["weight"] = position_embeddings

        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embeddings[i].clone()
        
    print("Converting QueryEmbedding layers...")
    query_embeddings = state_dict['module']['language_model']['topQueryEmbedding']['top_query_embeddings']['weight']
    out_query_embeddings = torch.chunk(query_embeddings, args.target_tensor_model_parallel_size, dim=0)
    
    for i in range(args.target_tensor_model_parallel_size):
        query_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.topQueryEmbedding.top_query_embeddings"
        )
        query_emb_dict["weight"] = out_query_embeddings[i].clone()
    
    print("Converting Transformer layers...")
    for layer_name in state_dict['module']['language_model']['transformer'].keys():
        params = state_dict['module']['language_model']['transformer'][layer_name]
        if "layernorm" in layer_name:
            pass
        elif "attention" in layer_name and "weight" in layer_name:
            if "dense" in layer_name:
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=1)
            else:
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
        elif "weight" in layer_name and "dense" in layer_name:
            if "h_to_4h" in layer_name:
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
            else:
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=1)
        elif "bias" in layer_name:
            if "dense" not in layer_name or "mlp" in layer_name:
                if "4h_to_h" in layer_name:
                    pass
                else:
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
                
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "module.language_model.transformer")
            if type(params) is tuple:
                params_dict[layer_name] = params[i].clone()
            else:
                params_dict[layer_name] = params
    
    os.makedirs(args.save_ckpt_path, exist_ok=True)
    for rank in range(args.target_tensor_model_parallel_size):
        save_ckpt_path = os.path.join(args.save_ckpt_path, f"mp_rank_{rank:02d}_model_states.pt")
        torch.save(output_state_dict[rank], save_ckpt_path)
        print(f"Converted checkpoint saved in {save_ckpt_path}.")


if __name__ == '__main__':
    main()

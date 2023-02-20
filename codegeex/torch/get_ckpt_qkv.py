import os
import sys
import torch
import random
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", 
                        type=str, 
                        default="/zhangpai24/workspace/ckpt_ms/ckpt_ms_213000_fp32_52224.pt")
    parser.add_argument("--save-path", 
                        type=str, 
                        default="/zhangpai24/workspace/ckpt_ms/ckpt_ms_213000_qkv.pt")
    
    args, _ = parser.parse_known_args()
    
    state_dict_path = args.load_path
    print("Loading state dict ...")
    sd = torch.load(state_dict_path, map_location="cpu")
    
    for i in range(40):
        if i < 39:
            query_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.query.weight', None)
            query_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.query.bias', None)
            key_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.key.weight', None)
            key_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.key.bias', None)
            value_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.value.weight', None)
            value_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.value.bias', None)
            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias])
            sd['module']['language_model']['transformer'][f'layers.{i}.attention.query_key_value.weight'] = qkv_weight
            sd['module']['language_model']['transformer'][f'layers.{i}.attention.query_key_value.bias'] = qkv_bias
        else:
            tq_key_weight = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.key.weight', None)
            tq_key_bias = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.key.bias', None)
            tq_value_weight = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.value.weight', None)
            tq_value_bias = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.value.bias', None)
            tq_kv_weight = torch.cat([tq_key_weight, tq_value_weight], dim=0)
            tq_kv_bias = torch.cat([tq_key_bias, tq_value_bias])
            sd['module']['language_model']['transformer']['topQueryLayer.attention.key_value.weight'] = tq_kv_weight
            sd['module']['language_model']['transformer']['topQueryLayer.attention.key_value.bias'] = tq_kv_bias
    
    save_ckpt_path = args.save_path
    torch.save(sd, save_ckpt_path)
    
if __name__ == '__main__':
    main()

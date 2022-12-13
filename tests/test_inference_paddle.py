
import os
import copy
import time
import paddle
import random
import argparse
import numpy as np

from codegeex.paddle.inference import get_token_stream
from codegeex.paddle import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer


def model_provider(args):
    """Build the model."""

    old_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype("float16")
    model = CodeGeeXModel(
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.padded_vocab_size,
        args.max_position_embeddings
    )
    model.language_model.embedding.word_embeddings.to(dtype="float32")
    model.language_model.embedding.position_embeddings.to(dtype="float32")
    model.language_model.topQueryEmbedding.top_query_embeddings.to(dtype="float32")
    for i in model.language_model.transformer.layers:
        i.input_layernorm.to(dtype="float32")
        i.post_attention_layernorm.to(dtype="float32")
    model.language_model.transformer.topQueryLayer.input_layernorm.to(dtype="float32")
    model.language_model.transformer.topQueryLayer.post_attention_layernorm.to(dtype="float32")
    model.language_model.transformer.final_layernorm.to(dtype="float32")
    paddle.set_default_dtype(old_dtype)
    
    return model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--num-layers",
        type=int,
        default=39,
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=5120,
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=40,
    )
    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=52224,
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy sampling.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        default="./test_prompt.txt",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizer",
    )
    group.add_argument(
        "--load",
        type=str,
    )
    group.add_argument(
        "--state-dict-path",
        type=str,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )
    
    return parser

    
def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    
    print("Loading tokenizer ...")
    tokenizer = CodeGeeXTokenizer(
        tokenizer_path=args.tokenizer_path, 
        mode="codegeex-13b")

    print("Loading state dict ...")
    state_dict = paddle.load(args.load)
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    model = model_provider(args)
    model.set_state_dict(state_dict)
    model.eval()
    model.to(dtype="float16")
    if args.quantize:
        raise NotImplementedError("quantize")
    
    with open(args.prompt_file, "r") as f:
        prompt = f.readlines()
        prompt = "".join(prompt)
    
    times = {}
    out_seq_lengths = [args.out_seq_length]
    micro_batch_size = args.micro_batch_size
    seq_length = args.max_position_embeddings
    for out_seq_length in out_seq_lengths:        
        print(f"Generating with out_seq_len {out_seq_length}...")
        
        times[out_seq_length] = []
        for prompt in [prompt]:
            t0 = time.perf_counter()
            tokens = tokenizer.encode_code(prompt)
            print(tokens)
            print("Current prompt:")
            print(prompt)
            n_token_prompt = len(tokens)
            print("N_token_prompt:", n_token_prompt)
            token_stream = get_token_stream(
                model,
                tokenizer,
                seq_length,
                out_seq_length,
                [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
                micro_batch_size=micro_batch_size,
                topk=args.top_k,
                topp=args.top_p,
                temperature=args.temperature,
                greedy=args.greedy,
            )
            is_finished = [False for _ in range(micro_batch_size)]
            for i, generated in enumerate(token_stream):
                generated_tokens = generated[0]
                for j in range(micro_batch_size):
                    if is_finished[j]:
                        continue
                    if generated_tokens[j].cpu().numpy()[-1] == tokenizer.eos_token_id or len(
                            generated_tokens[j]) >= out_seq_length:
                        is_finished[j] = True
                        generated_tokens_ = generated_tokens[j].cpu().numpy().tolist()
                        generated_code = tokenizer.decode_code(generated_tokens_[n_token_prompt:])
                        generated_code = "".join(generated_code)
                        t1 = time.perf_counter()
                        print("Total generation time:", t1 - t0, "# Tokens:", len(generated_tokens_) - n_token_prompt)
                        print(f"{(t1 - t0) / (len(generated_tokens_) - n_token_prompt)}s/token")
                        times[out_seq_length].append(t1 - t0)
                        print("================================= Generated code:")
                        print(generated_code)
                        
                    if all(is_finished):
                        break
                    
    print(times)
    for out_seq_length in times.keys():
        print(out_seq_length, np.mean(times[out_seq_length]))
        
    print("Generation finished.")


if __name__ == "__main__":
    main()

import copy

from typing import *
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.torch.inference import get_token_stream


def get_model(
    backend: str = "megatron",
    quantized: bool = False,
):
    pass


def generate(
    model, 
    tokenizer: CodeGeeXTokenizer, 
    prompt: str, 
    out_seq_length: int,
    seq_length: int = 2048,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    micro_batch_size: int = 1,
    backend: str = "megatron",
    greedy: bool = False,
    verbose: bool = False,
):
    tokens = tokenizer.encode_code(prompt)
    n_token_prompt = len(tokens)

    if verbose:
        print(f"Current prompt:\n{prompt}")
        print("N_token_prompt:", n_token_prompt)
    
    generated_codes = []
    if backend == "megatron":
        token_stream = get_token_stream(
            model,
            tokenizer,
            seq_length,
            out_seq_length,
            [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
            micro_batch_size=micro_batch_size,
            topk=top_k,
            topp=top_p,
            temperature=temperature,
            greedy=greedy,
        )
        is_finished = [False for _ in range(micro_batch_size)]
        for i, generated in enumerate(token_stream):
            generated_tokens = generated[0]
            for j in range(micro_batch_size):
                if is_finished[j]:
                    continue
                
                if generated_tokens[j].cpu().numpy()[-1] == tokenizer.eos_token_id or len(generated_tokens[j]) >= out_seq_length:
                    is_finished[j] = True
                    generated_tokens_ = generated_tokens[j].cpu().numpy().tolist()
                    generated_code = tokenizer.decode_code(generated_tokens_[n_token_prompt:])
                    generated_code = "".join(generated_code)
                    generated_codes.append(generated_code)
                    if verbose:
                        print(f"\nGenerated code {i}:\n{generated_code}")
                    
                if all(is_finished):
                    break

    return generated_codes
import copy
import json
import random
import traceback
from typing import *

import numpy
import torch
import zmq

from codegeex.benchmark.utils import is_code_generation_finished, cleanup_code
from codegeex.megatron import get_args, get_tokenizer
from codegeex.megatron import mpu
from codegeex.megatron.code_generation_utils import get_token_stream
from codegeex.megatron.model import CodeGeeXModel


def model_provider():
    """Build the model."""

    model = CodeGeeXModel(num_tokentypes=0,
                          parallel_output=False)

    return model


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    mpu.model_parallel_cuda_manual_seed(seed)


def run_generation_distributed(model):
    args = get_args()
    if hasattr(args, "language_tgt_type"):
        language_type = args.language_tgt_type
    else:
        language_type = args.language_type
    print(f"Connecting to tcp://{args.channel_ip}:{args.channel_port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.channel_ip}:{args.channel_port}")
    output_file_path = args.output_prefix + f"_finished_rank{args.gen_rank}.jsonl"
    unfinished_output_file_path = args.output_prefix + f"_unfinished_rank{args.gen_rank}.jsonl"
    problems = {}
    print("Building tokenizer...")
    tokenizer = get_tokenizer()

    with open(output_file_path, "w") as f:
        with open(unfinished_output_file_path, "w") as unfinished_f:
            while True:
                socket.send_json({"rank": args.gen_rank, "action": "pull"})
                resp = socket.recv_json()
                try:
                    if "codecontest" in args.dataset.lower():
                        if resp["contest_name"] is None:
                            break
                    elif resp["task_id"] is None:
                        break

                    if "codecontest" in args.dataset.lower():
                        current_spec = problems[resp["contest_name"]]
                        prompt = current_spec.prompt
                    else:
                        current_spec = resp["task_id"]
                        prompt = current_spec["prompt"]

                    temperature = None if "temperature" not in resp else resp["temperature"]
                    topp = None if "topp" not in resp else resp["topp"]

                    f.flush()
                    unfinished_f.flush()
                    tokens = tokenizer.tokenize(prompt)
                    n_token_prompt = len(tokens)
                    if n_token_prompt >= args.seq_length:
                        continue
                    if "micro_batch_size" in resp:
                        micro_batch_size = resp["micro_batch_size"]
                    else:
                        micro_batch_size = args.micro_batch_size
                    if args.beam_search:
                        beams = get_token_stream(
                            model,
                            [
                                copy.deepcopy(tokens)
                                for _ in range(micro_batch_size)
                            ],
                            return_scores=args.return_scores,
                            prompt_length=n_token_prompt,
                            micro_batch_size=micro_batch_size,
                            bad_ids=args.bad_ids,
                            temperature=temperature,
                            topp=topp,
                            beam_warmup=args.beam_warmup,
                        )
                        for beam in beams:
                            generated_tokens_ = beam.tokens
                            generated_tokens_ = (
                                generated_tokens_
                                if generated_tokens_[-1] != tokenizer.eod
                                else generated_tokens_[:-1]
                            )
                            generated_code = tokenizer.detokenize(generated_tokens_[n_token_prompt:])
                            generated_code = cleanup_code(generated_code,
                                                          language_type=language_type,
                                                          dataset=args.dataset)
                            f.write(
                                json.dumps(
                                    {
                                        "task_id"   : current_spec['task_id'],
                                        "prompt"    : prompt,
                                        "generation": generated_code,
                                        "scores"    : beam.score,
                                        "finish"    : 2 if generated_tokens[i].cpu().numpy()[
                                                               -1] == tokenizer.eod else 1,
                                        "output"    : beam.tokens,
                                    }
                                )
                                + "\n"
                            )
                        socket.send_json(
                            {
                                "rank"   : args.gen_rank,
                                "action" : "success",
                                "task_id": current_spec['task_id']
                            }
                        )
                        socket.recv()
                        continue

                    token_stream = get_token_stream(
                        model,
                        [
                            copy.deepcopy(tokens)
                            for _ in range(micro_batch_size)
                        ],
                        return_scores=args.return_scores,
                        prompt_length=n_token_prompt,
                        micro_batch_size=micro_batch_size,
                        bad_ids=args.bad_ids,
                        temperature=temperature,
                        topp=topp,
                        beam_warmup=args.beam_warmup,
                    )
                    is_finished = [False for _ in range(micro_batch_size)]
                    for generated in token_stream:
                        generated_tokens = generated[0]
                        if args.return_scores:
                            scores = generated[1][1]
                        else:
                            scores = None

                        for i in range(micro_batch_size):
                            if is_finished[i]:
                                continue

                            generated_tokens_ = generated_tokens[i].cpu().numpy().tolist()
                            generated_tokens_ = (
                                generated_tokens_
                                if generated_tokens_[-1] != tokenizer.eod
                                else generated_tokens_[:-1]
                            )
                            generated_code = tokenizer.detokenize(generated_tokens_[n_token_prompt:])
                            if generated_tokens[i].cpu().numpy()[-1] == tokenizer.eod or \
                                    is_code_generation_finished(
                                        generated_code,
                                        language_type=language_type,
                                        dataset=args.dataset,
                                    ):
                                is_finished[i] = True
                                generated_code = cleanup_code(generated_code,
                                                              language_type=language_type,
                                                              dataset=args.dataset)
                                f.write(
                                    json.dumps(
                                        {
                                            "task_id"   : current_spec['task_id'],
                                            "prompt"    : prompt,
                                            "generation": generated_code,
                                            "scores"    : 0.0 if scores is None else scores[i].detach().cpu().item(),
                                            "finish"    : 2 if generated_tokens[i].cpu().numpy()[
                                                                   -1] == tokenizer.eod else 1,
                                            "output"    : generated_tokens[i].cpu().numpy().tolist(),
                                        }
                                    )
                                    + "\n"
                                )

                            if len(generated_tokens[i]) >= args.out_seq_length:
                                break

                        if all(is_finished):
                            break

                    for i in range(micro_batch_size):
                        if not is_finished[i]:
                            generated_tokens_ = generated_tokens[i].cpu().numpy().tolist()
                            generated_code = tokenizer.detokenize(generated_tokens_[n_token_prompt:])
                            unfinished_f.write(
                                json.dumps(
                                    {
                                        "task_id"   : current_spec['task_id'],
                                        "prompt"    : prompt,
                                        "generation": generated_code,
                                        "scores"    : 0.0 if scores is None else scores[i].detach().cpu().item(),
                                        "finish"    : 0,
                                        "output"    : generated_tokens_,
                                    }
                                )
                                + "\n"
                            )

                    socket.send_json(
                        {
                            "rank"   : args.gen_rank,
                            "action" : "success",
                            "task_id": current_spec['task_id']
                        }
                    )
                    socket.recv()

                except Exception as e:
                    print(f"*** (rank={args.gen_rank}) crashed.")
                    print(f"    error: {repr(e)}")
                    traceback.print_exc()
                    if args.dataset.lower() == "codecontest":
                        socket.send_json({
                            "rank"            : args.gen_rank,
                            "action"          : "fail",
                            "contest_name"    : current_spec.name,
                            "micro_batch_size": micro_batch_size
                        })
                    else:
                        socket.send_json(
                            {
                                "rank"   : args.gen_rank,
                                "action" : "fail",
                                "task_id": current_spec['task_id']
                            }
                        )
                    socket.recv()
                    continue

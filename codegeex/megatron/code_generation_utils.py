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

"""Utilities for generating text."""

import copy
import json
import os
import time
from typing import *

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from codegeex.megatron import get_args, print_rank_0
from codegeex.megatron import get_tokenizer
from codegeex.megatron import mpu
from codegeex.megatron.utils import get_ltor_masks_and_position_ids
from codegeex.benchmark.utils import is_code_generation_finished


def get_batch(context_tokens, micro_batch_size=None):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    if micro_batch_size is None:
        micro_batch_size = args.micro_batch_size
    tokens = context_tokens.view(micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, attention_mask, position_ids


def get_batch_(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate_samples_input_from_file(model):
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, "sample input file is not provided."
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print(
                "`sample-output-file` not specified, setting "
                "it to {}".format(sample_output_file)
            )
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if (
                    mpu.is_pipeline_first_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print(
                            "\nContext length",
                            context_length,
                            "\nPlease give smaller context (half of the "
                            "sequence length)!",
                            flush=True,
                        )
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(
                input_info_tensor, group=mpu.get_model_parallel_group()
            )
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and args.pipeline_model_parallel_size > 1
            ):
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(
                        context_length, dtype=torch.int64, device=torch.device("cuda")
                    )
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass

            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():
                    os.system("clear")
                    print("\nContext:", raw_text, flush=True)

                    fname_out.write("\nContext:")
                    fname_out.write(raw_text)

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(decode_tokens)[
                                         raw_text_len:
                                         ]
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                    fname_out.write("\n\nMegatron-LM:")
                    fname_out.write(trim_decode_tokens)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1


# We added this function to support the tasks evaluation such as squad
# and drop in the https://github.com/EleutherAI/lm-evaluation-harness
# codebase. The lm-evaluation-harness code can now call this function
# similar to their current generate function call used for gpt style models.
def generate_samples_eval(model, context, max_gen_length, eos_token_id):
    # Generate samples for lm evaluation
    # NEED TO THINK ABOUT eos token

    args = get_args()
    tokenizer = get_tokenizer()

    raw_text_len = len(context)
    model.eval()

    context_tokens = tokenizer.tokenize(context)
    args.out_seq_length = max_gen_length + len(context_tokens)
    args.eos_id = eos_token_id

    with torch.no_grad():
        token_stream = get_token_stream(model, [context_tokens])
        for counter, decode_tokens in enumerate(token_stream):
            if counter == args.out_seq_length:
                break

    decode_tokens, _ = decode_tokens
    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
    trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]

    return trim_decode_tokens


def generate_samples_interactive_code_contest(model, print_frequency=10):
    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if (
                    mpu.is_pipeline_first_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                # os.system("clear")
                raw_text = []
                input_line = input("\nContext prompt (EOF to exit) >>> ")

                if input_line == ":recompute":
                    args.recompute = True
                    print(f"set recompute: {args.recompute}")
                    continue

                if input_line == ":no-recompute":
                    args.recompute = False
                    print(f"set recompute: {args.recompute}")
                    continue

                while input_line != "EOF":
                    raw_text.append(input_line)
                    input_line = input("\nContext prompt (EOF to exit) >>> ")
                raw_text = "\n".join(raw_text)

                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    # terminate_runs = 1
                    pass
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print(
                            "\nContext length",
                            context_length,
                            "\nPlease give smaller context (half of the "
                            "sequence length)!",
                            flush=True,
                        )
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(
                input_info_tensor, group=mpu.get_model_parallel_group()
            )
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and args.pipeline_model_parallel_size > 1
            ):
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(
                        context_length, dtype=torch.int64, device=torch.device("cuda")
                    )
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens for _ in range(args.micro_batch_size)])

            for counter, decode_tokens in enumerate(token_stream):
                if (
                        counter % print_frequency != 0
                        or mpu.get_tensor_model_parallel_rank() != 0
                        or not mpu.is_pipeline_first_stage()
                ):
                    continue

                os.system("clear")
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]
                print(f"\nMegatron-LM (gen len: {counter}):", trim_decode_tokens, flush=True)

            if (
                    mpu.is_pipeline_first_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                os.system("clear")
                print("\nContext:", raw_text, flush=True)

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1


def generate_samples_interactive(model, print_frequency=24):
    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            terminate_runs = 0
            raw_text_len = 0

            if (
                    mpu.is_pipeline_first_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                os.system("clear")
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print("Prompt should not be empty!")
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
                raw_text_len = len(raw_text)

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print(
                            "\nContext length",
                            context_length,
                            "\nPlease give smaller context (half of the "
                            "sequence length)!",
                            flush=True,
                        )
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = 0

            input_info = [terminate_runs, raw_text_len, context_length]
            input_info_tensor = torch.cuda.LongTensor(input_info)
            torch.distributed.all_reduce(
                input_info_tensor, group=mpu.get_model_parallel_group()
            )
            terminate_runs = input_info_tensor[0].item()
            raw_text_len = input_info_tensor[1].item()
            context_length = input_info_tensor[2].item()

            if terminate_runs == 1:
                return

            # For pipeline parallel we send context tokens to other stages
            # so they get the lengths correct
            if (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and args.pipeline_model_parallel_size > 1
            ):
                if mpu.is_pipeline_first_stage():
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                else:
                    src = mpu.get_pipeline_model_parallel_first_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    context_tokens_tensor = torch.empty(
                        context_length, dtype=torch.int64, device=torch.device("cuda")
                    )
                    torch.distributed.broadcast(context_tokens_tensor, src, group)
                    context_tokens = context_tokens_tensor.cpu().numpy().tolist()

            token_stream = get_token_stream(model, [context_tokens])

            for counter, decode_tokens in enumerate(token_stream):
                if (
                        counter % print_frequency != 0
                        or mpu.get_tensor_model_parallel_rank() != 0
                        or not mpu.is_pipeline_first_stage()
                ):
                    continue

                os.system("clear")
                print("\nContext:", raw_text, flush=True)

                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if (
                    mpu.is_pipeline_first_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                os.system("clear")
                print("\nContext:", raw_text, flush=True)

                if not isinstance(decode_tokens, list):
                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                input("\nPress Enter to continue >>>")

            raw_text = None
            context_count += 1


def generate_samples_unconditional(model):
    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod] for _ in range(args.micro_batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        for token_stream in get_token_stream(model, copy.deepcopy(context_tokens)):
            pass
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            if ctr % args.log_interval == 0:
                print(
                    "Avg s/batch:",
                    (time.time() - start_time) / min(args.log_interval, ctr + 1),
                )
                start_time = time.time()
            length = len(token_stream)
            token_batch = token_stream[0].cpu().numpy().tolist()
            length_batch = token_stream[1].cpu().numpy().tolist()
            assert len(length_batch) == args.micro_batch_size
            for tokens, length in zip(token_batch, length_batch):
                tokens = tokens[1: length - 1]
                text = tokenizer.detokenize(tokens)
                is_finished = length < args.seq_length - 1
                datum = {"text": text, "length": length - 1, "finished": is_finished}
                yield datum
                ctr += 1
                if ctr >= num_samples:
                    break
        else:
            for _ in range(args.micro_batch_size):
                yield None
                ctr += 1
                if ctr >= num_samples:
                    break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model):
    args = get_args()
    assert args.genfile is not None
    with open(args.genfile, "w") as f:
        for datum in generate_samples_unconditional(model):
            if (
                    mpu.is_pipeline_last_stage()
                    and mpu.get_tensor_model_parallel_rank() == 0
            ):
                f.write(json.dumps(datum) + "\n")


def pad_batch(batch, pad_id, args):
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def topk_sampling(logits: torch.FloatTensor, num_samples: int):
    """
    Samples from a multinomial distribution using the top-k sampling strategy.

    Args:
        logits: A tensor of shape (batch_size, vocab_size) containing the logits.
        num_samples: The number of samples to draw.
    """
    log_prob = F.log_softmax(logits, dim=-1)
    topk = torch.topk(log_prob, num_samples, dim=-1)
    topk_tokens = topk.indices
    topk_log_prob = topk.values

    return topk_tokens, topk_log_prob


def nuclear_sampling(logits: torch.FloatTensor, temperature: float, top_p: float = None, top_k: int = None):
    orig_log_probs = F.log_softmax(logits, dim=-1)
    logits /= temperature
    logits = top_k_logits(logits, top_k, top_p)
    log_probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(log_probs, num_samples=1).view(-1)

    indices = tokens.view(-1, 1)
    new_scores = orig_log_probs.gather(1, indices).view(-1)

    return tokens, new_scores


def sample_topk_tokens(model,
                       input_tokens, attention_mask, position_ids,
                       context_length: int, num_samples: int):
    assert context_length < input_tokens.shape[-1], "context_length must be smaller than seq_length"

    model.eval()
    with torch.no_grad():
        output = forward_step(
            model,
            input_tokens,
            position_ids,
            attention_mask,
            tokentype_ids=None,
            forward_method_parallel_output=False,
        )
    assert output is not None
    logits = output[:, context_length - 1, :]

    return topk_sampling(logits, num_samples)


def nuclear_sample_tokens(model,
                          input_tokens, attention_mask, position_ids,
                          context_length: int, temperature: float, top_p: float, top_k: int):
    assert context_length < input_tokens.shape[-1], "context_length must be smaller than seq_length"

    model.eval()
    with torch.no_grad():
        output = forward_step(
            model,
            input_tokens,
            position_ids,
            attention_mask,
            tokentype_ids=None,
            forward_method_parallel_output=False,
        )
    assert output is not None
    logits = output[:, context_length - 1, :]
    return nuclear_sampling(logits, temperature, top_p, top_k)


@dataclass
class Beam:
    tokens: List[int]
    score: float

    def __repr__(self):
        return f"<Beam {repr(get_tokenizer().detokenize(self.tokens))}, score={self.score}>"

    def get_code(self):
        return get_tokenizer().detokenize(self.tokens)


def expand_beams(beams: List[Beam], num_beams: int, model) -> List[Beam]:
    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens = [b.tokens.copy() for b in beams]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eod, args)

    context_lengths = set(context_lengths)
    assert len(context_lengths) == 1, "context_lengths must be the same"
    context_length = list(context_lengths)[0]

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    tokens, attention_mask, position_ids = get_batch_(context_tokens_tensor)
    tokens, scores = sample_topk_tokens(model, tokens, attention_mask, position_ids, context_length, num_beams)
    tokens = tokens.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()
    assert len(tokens) == len(beams), "output tokens and input beams must have the same length"

    all_beams = []
    for i in range(len(beams)):
        this_tokens = tokens[i]
        this_scores = scores[i]

        for token, score in zip(this_tokens, this_scores):
            all_beams.append(Beam(beams[i].tokens + [token], beams[i].score + score))

    return all_beams


def beam_search(model, context_tokens, num_beams: int):
    """Beam search.

    Note that this function does not support model parallel!
    """
    args = get_args()
    tokenizer = get_tokenizer()

    assert not isinstance(context_tokens[0], list), "batched beam search not supported"

    initial_beam = Beam(context_tokens, 0.0)
    context_len = len(context_tokens)
    org_context_len = context_len
    finished_beams = []

    # first expansion
    beams = expand_beams([initial_beam], num_beams, model)
    context_len += 1

    # print(f"initial beam: {initial_beam}")

    while len(beams) > 0 and context_len < args.seq_length:
        expanded_beams = expand_beams(beams, num_beams, model)
        next_beams = []
        for beam in expanded_beams:
            if args.beam_warmup:
                if len(beam.tokens) >= org_context_len + args.beam_warmup_length or beam.tokens[-1] == tokenizer.eod:
                    finished_beams.append(beam)
                else:
                    next_beams.append(beam)
            else:
                if args.evaluation:
                    generated_code = tokenizer.detokenize(beam.tokens[org_context_len:])
                    if is_code_generation_finished(generated_code):
                        finished_beams.append(beam)
                        continue
                if beam.tokens[-1] == tokenizer.eod:
                    finished_beams.append(beam)
                else:
                    next_beams.append(beam)
        # only keep top-k beams
        next_beams.sort(key=lambda b: b.score, reverse=True)
        beams = next_beams[:num_beams]
        context_len += 1

        if len(finished_beams) >= num_beams:
            # first, only keep top-k beams
            finished_beams.sort(key=lambda b: b.score, reverse=True)
            finished_beams = finished_beams[:num_beams]
            return finished_beams  # return finished beams with highest scores
            # stop if all currently expanding beams has a score lower than the minimal score of finished ones
            min_score = min([b.score for b in finished_beams])
            if min_score >= beams[0].score:
                break
            else:
                print(f"we have got enough finished beams, but the minimal score is {min_score}")
                print(f"and the maximum searching score is {beams[0].score}")

    # return top-k finished and unfinished beams
    all_beams = finished_beams + beams
    all_beams.sort(key=lambda b: b.score, reverse=True)

    return all_beams[:num_beams]


@dataclass
class Handle:
    tokens: List[int]
    score: float

    def __repr__(self):
        return f"<Handle {repr(get_tokenizer().detokenize(self.tokens))}, score={self.score}>"

    def is_finished(self):
        return len(self.tokens) and self.tokens[-1] == get_tokenizer().eod

    def derived(self, new_token: int, log_prob: float):
        assert not self.is_finished(), "cannot derive from a finished handle"
        return Handle(self.tokens + [new_token], self.score + log_prob)


def expand_handles(handles: List[Handle], temperature: float, top_p: float, top_k: int, model):
    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens = [b.tokens.copy() for b in handles]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eod, args)

    context_lengths = set(context_lengths)
    assert len(context_lengths) == 1, "context_lengths must be the same"
    context_length = list(context_lengths)[0]

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    tokens, attention_mask, position_ids = get_batch_(context_tokens_tensor)
    tokens, scores = nuclear_sample_tokens(model, tokens, attention_mask, position_ids, context_length, temperature,
                                           top_p, top_k)
    tokens = tokens.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()
    assert len(tokens) == len(handles), "output tokens and input must have the same length"

    all_beams = []
    for i in range(len(handles)):
        this_tokens = tokens[i]
        this_scores = scores[i]

        all_beams.append(handles[i].derived(this_tokens, this_scores))

    return all_beams


def generate_nuclear_sampling(model, context_tokens, num_samples: int, temperature: float, top_p: float, top_k: int):
    """Beam search.

    Note that this function does not support model parallel!
    """
    args = get_args()
    tokenizer = get_tokenizer()

    assert not isinstance(context_tokens[0], list), "batched beam search not supported"

    handles = [Handle(tokens=context_tokens, score=0) for _ in range(num_samples)]
    context_len = len(context_tokens)
    finished_handles = []

    while len(handles) > 0 and context_len < args.seq_length:
        expanded_handles = expand_handles(handles, temperature, top_p, top_k, model)

        new_handles = []
        for h in expanded_handles:
            if h.is_finished():
                finished_handles.append(h)
            else:
                new_handles.append(h)

        context_len += 1
        handles = new_handles

    return handles + finished_handles


def forward_step(
        model,
        tokens,
        position_ids,
        attention_mask,
        tokentype_ids,
        layer_past=None,
        get_key_value=None,
        forward_method_parallel_output=None,
        prompt_length=None,
        context_length=None,
):
    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size
    args = get_args()
    orig_seq_length = args.seq_length
    args.seq_length = tokens.shape[1]

    # Forward pass through the model.
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        tokentype_ids=tokentype_ids,
        layer_past=layer_past,
        get_key_value=get_key_value,
        prompt_length=prompt_length,
        context_length=context_length,
    )

    if get_key_value:
        output_tensor, layer_past = output_tensor

    args.seq_length = orig_seq_length
    if get_key_value:
        return output_tensor, layer_past

    return output_tensor


def get_token_stream(
        model,
        context_tokens,
        return_scores: bool = False,
        prompt_length: int = None,
        micro_batch_size: int = None,
        bad_ids: List = None,
        temperature: float = None,
        topp: float = None,
        topk: int = None,
):
    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(
        context_length_tensor,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    torch.distributed.broadcast(
        context_tokens_tensor,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, micro_batch_size)

    batch_token_iterator = sample_sequence_batch(
        model,
        context_tokens_tensor,
        context_length_tensor,
        attention_mask,
        position_ids,
        return_scores=return_scores,
        prompt_length=prompt_length,
        bad_ids=bad_ids,
        temperature=temperature,
        topp=topp,
        topk=topk,
    )

    if args.beam_search:
        for beams in batch_token_iterator:
            yield beams
    else:
        for tokens, lengths in batch_token_iterator:
            context_length += 1
            if tokens is not None:
                yield tokens[:, :context_length], lengths
            else:
                yield None, None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
        model,
        context_tokens,
        context_lengths,
        attention_mask,
        position_ids,
        maxlen=None,
        type_ids=None,
        return_scores: bool = False,
        prompt_length: int = None,
        bad_ids: List = None,
        temperature: float = None,
        topp: float = None,
        topk: int = None,
):
    args = get_args()
    tokenizer = get_tokenizer()
    temperature = temperature if temperature is not None else args.temperature
    topp = topp if topp is not None else args.top_p
    topk = topk if topk is not None else args.top_k

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        if hasattr(args, "eos_id"):
            eos_id = args.eos_id
        else:
            eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        if return_scores:
            scores = torch.zeros([batch_size]).float().cuda()

        if args.beam_search:
            beams = beam_search(model, context_tokens=tokens.cpu().numpy().tolist()[0][:context_length],
                                num_beams=args.num_beams)
            if args.beam_warmup:
                beam = beams[0]
                tokens_ = beam.tokens
                tokens_ = (tokens_ if tokens_[-1] != tokenizer.eod else tokens_[:-1])
                tokens_warmup = []
                for i in range(batch_size):
                    tokens_warmup.append(tokens_.copy())
                tokens, context_lengths = pad_batch(tokens_warmup, tokenizer.eod, args)
                tokens = torch.cuda.LongTensor(tokens)
                context_lengths = torch.cuda.LongTensor(context_lengths)
                context_length = len(tokens_)
                org_context_length = context_length
                if maxlen is None:
                    maxlen = args.seq_length - 1
                    if maxlen > (org_context_length + args.out_seq_length):
                        maxlen = org_context_length + args.out_seq_length
                lengths = torch.ones([batch_size]).long().cuda() * maxlen
                tokens, attention_mask, position_ids = get_batch(tokens, batch_size)
            else:
                yield beams
        else:
            while context_length <= (maxlen):
                if args.recompute:
                    logits = model(tokens,
                                position_ids,
                                attention_mask,
                                tokentype_ids=type_ids,
                                forward_method_parallel_output=False,
                                prompt_length=prompt_length,
                                context_length=context_length,
                                )
                    logits = logits[:, context_length - 1, :]
                else:
                    types2use = None
                    if counter == 0:
                        tokens2use = tokens[:, :context_length]
                        positions2use = position_ids[:, :context_length]
                        if type_ids is not None:
                            types2use = type_ids[:, :context_length]
                    else:
                        tokens2use = tokens[:, context_length - 1].view(
                            batch_size, -1)
                        positions2use = position_ids[:, context_length - 1].view(
                            batch_size, -1)
                        if type_ids is not None:
                            types2use = type_ids[:, context_length - 1].view(
                                batch_size, -1)
                    logits, layer_past = model(tokens2use,
                                            positions2use,
                                            attention_mask,
                                            layer_past=layer_past,
                                            get_key_value=True,
                                            tokentype_ids=types2use,
                                            forward_method_parallel_output=False,
                                            prompt_length=prompt_length,
                                            context_length=context_length,
                                            )
                    logits = logits[:, -1].view(batch_size, -1).contiguous()

                if mpu.is_pipeline_last_stage():
                    if bad_ids is not None:
                        for bad_id in bad_ids:
                            logits[:, bad_id] = -10000
                    if args.greedy:
                        prev = torch.argmax(logits, dim=-1).view(-1)
                    else:
                        logits = logits.float()
                        if return_scores:
                            orig_log_probs = torch.log_softmax(logits, dim=-1)
                        logits /= temperature
                        logits = top_k_logits(logits, top_k=topk, top_p=topp)
                        log_probs = F.softmax(logits, dim=-1)
                        prev = torch.multinomial(log_probs, num_samples=1).view(-1)

                    started = context_lengths <= context_length

                    new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                    if not args.greedy and return_scores:
                        indices = prev.view(-1, 1)
                        new_scores = orig_log_probs.gather(1, indices).view(-1)
                        new_scores = new_scores * started
                        new_scores = new_scores * is_done.bool().logical_not()
                        scores += new_scores

                    tokens[:, context_length] = new_tokens
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_embedding_group()
                    torch.distributed.broadcast(new_tokens, src, group)

                    done_token = (prev == eos_id).byte() & started.byte()
                    just_finished = (done_token & ~is_done).bool()
                    lengths[just_finished.view(-1)] = context_length
                    is_done = is_done | done_token

                    done = torch.all(is_done)
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    torch.distributed.broadcast(done, src, group)

                    if return_scores:
                        yield tokens, (lengths, scores)
                    else:
                        yield tokens, lengths

                else:
                    if mpu.is_pipeline_first_stage():
                        src = mpu.get_pipeline_model_parallel_last_rank()
                        group = mpu.get_embedding_group()
                        new_tokens = torch.empty_like(tokens[:, context_length])
                        torch.distributed.broadcast(new_tokens, src, group)
                        tokens[:, context_length] = new_tokens
                        yield tokens, None
                    else:
                        yield None, None

                    done = torch.cuda.ByteTensor([0])
                    src = mpu.get_pipeline_model_parallel_last_rank()
                    group = mpu.get_pipeline_model_parallel_group()
                    torch.distributed.broadcast(done, src, group)

                context_length += 1
                counter += 1
                if done:
                    break

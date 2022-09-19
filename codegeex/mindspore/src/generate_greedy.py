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

"""
TopK for text generation
"""

from copy import deepcopy

import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor


def topk_fun(logits, topk=5):
    """Get topk"""
    value = np.flip(np.sort(logits), axis=-1)[..., :topk]
    index = np.flip(np.argsort(logits), axis=-1)[..., :topk]
    return value, index


def sampler(log_probs_revised, top_p, top_k_num, use_pynative=False, bad_words_index=[]):
    for i, bad_words in enumerate(bad_words_index):
        for bad_word in bad_words:
            log_probs_revised[i, bad_word] = -10000
    """Convert the log_probs to probability"""
    if use_pynative:
        log_probs_revised = log_probs_revised.asnumpy()

    return log_probs_revised.argmax(axis=1)


def generate_increment(model, origin_inputs, origin_length, config, tokenizer, verbose=False):
    """
    Text generation for incremental inference

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        config: inference configurations

    Returns:
        outputs: the ids for the generated text
    """
    # Get configurations for inference
    frequency_penalty = config.frequency_penalty
    presence_penalty = config.presence_penalty
    top_p = config.top_p
    top_k_num = config.top_k_num
    temperature = config.temperature
    max_generate_length = config.max_generate_length
    seq_length = config.seq_length
    end_token = config.end_token
    use_pynative = config.use_pynative_op
    vocab_embedding_vocab_size = (config.vocab_size // 1024 + 1) * 1024

    batch_size, valid_length = origin_inputs.shape
    # Init outputs with original inputs
    outputs = [[origin_inputs[i][j] for j in range(valid_length)] for i in range(batch_size)]
    output_codes = [[] for _ in range(batch_size)]
    # If target length exceeds seq_length, use seq_length instead
    target_lengths = [min(seq_length, l + max_generate_length) for l in origin_length]
    valid_lengths = deepcopy(origin_length)
    gen_end = [(l == -1) for l in origin_length]

    # A list of the frequency of each token
    frequency_list = np.zeros((batch_size, vocab_embedding_vocab_size))
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)),
                       'constant', constant_values=(end_token, end_token))
    if verbose:
        print("input_ids is ", input_ids)

    # Indicate the exact token position
    current_indexes = [max(l - 1, 0) for l in valid_lengths]
    # batch_valid_length = Tensor(np.array([current_index for _ in range(batch_size)]), mstype.int32)
    batch_valid_length = Tensor(np.array(current_indexes), mstype.int32)
    current_indexes = Tensor(np.array([current_indexes[i] + i * seq_length for i in range(batch_size)]), mstype.int32)
    # For first graph, not_init should be false
    init_true = Tensor([True], mstype.bool_)
    init_false = Tensor([False], mstype.bool_)
    init = init_false
    # Claim the first graph
    model.predict_network.add_flags_recursive(is_first_iteration=True)
    # Call a single inference with input size of (bs, seq_length)
    logits = model.predict(Tensor(input_ids, mstype.int32),
                           current_indexes, init, batch_valid_length)

    # Claim the second graph and set not_init to true
    init = init_true
    model.predict_network.add_flags_recursive(is_first_iteration=False)

    comments_index = [2, ]  # '#': 2, ' #': 1303
    newline_index = [198, ]  # '\n': 198
    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while not all(gen_end):
        # Reshape the output logits
        logits = logits.asnumpy()
        log_probs = logits.reshape(batch_size, vocab_embedding_vocab_size)

        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty

        bad_words_index = [[] for _ in range(batch_size)]

        target_index = sampler(log_probs_revised, top_p, top_k_num, use_pynative, bad_words_index=bad_words_index)

        if verbose:
            print(f"=== Length {valid_lengths}, target index {target_index}, generation end status {gen_end}.")

        # Update frequency list
        target = target_index
        frequency_list[np.arange(batch_size), target] = frequency_list[np.arange(batch_size), target] + 1

        batch_valid_length = Tensor(np.array(valid_lengths), mstype.int32)
        current_indexes = Tensor(np.arange(batch_size, dtype=np.int32), mstype.int32)
        input_id = Tensor([target], mstype.int32).reshape(-1, 1)
        # Update outputs with current generated token
        for i in range(batch_size):
            if not gen_end[i]:
                if int(target[i]) == 50256:
                    gen_end[i] = True
                else:
                    output_codes[i].append(int(target[i]))
                    outputs[i].append(int(target[i]))
                    valid_lengths[i] += 1
                if valid_lengths[i] >= target_lengths[i]:
                    gen_end[i] = True

        # Call a single inference with input size of (bs, 1)
        logits = model.predict(input_id, current_indexes,
                               init, batch_valid_length)
    return tokenizer.decode_code(output_codes)

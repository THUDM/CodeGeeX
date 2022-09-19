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
import mindspore.common.dtype as mstype
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P


def is_code_generation_finished(text: str):
    """
    Checks whether the generated code text is finished.
    """
    # end_words = ['\ndef', '\nclass', '\nif', '\n#', '\nprint', '<|endoftext|>']
    end_words = ['\n}']
    for w in end_words:
        if w in text:
            return True
    return False


def cleanup_text(text: str):
    """
    Cleans up the generated code text.
    """
    # end_words = ['\ndef', '\nclass', '\nif', '\n#', '\nprint', '<|endoftext|>']
    end_words = ['\n}']
    for w in end_words:
        if text.endswith(w):
            text = text[:-len(w)]
    return text


def truncate_text(text: str):
    """
    Cleans up the generated code text.
    """
    # end_words = ['\ndef', '\nclass', '\nif', '\n#', '\nprint', '<|endoftext|>']
    end_words = ['\n}']
    for w in end_words:
        idx = text.find(w)
        if idx != -1:
            text = text[:idx] + w
            # text = text[:idx]
    return text


def topk_fun(logits, topk=5):
    """Get topk"""
    # target_column = logits[0].tolist()
    # sorted_array = [(k, v) for k, v in enumerate(target_column)]
    # sorted_array.sort(key=lambda x: x[1], reverse=True)
    # topk_array = sorted_array[:topk]
    # index, value = zip(*topk_array)
    # index = np.array([index])
    # value = np.array([value])
    value = np.flip(np.sort(logits), axis=-1)[..., :topk]
    index = np.flip(np.argsort(logits), axis=-1)[..., :topk]
    return value, index


def sampler(log_probs_revised, top_p, top_k_num, use_pynative=False, bad_words_index=[]):
    for i, bad_words in enumerate(bad_words_index):
        for bad_word in bad_words:
            log_probs_revised[i, bad_word] = -10000
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(np.e, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(np.e, np.array(log_probs_revised, np.float32))

        # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)

        sorted_p = sorted_logits / sorted_logits.sum(axis=1).reshape(-1, 1)
        cumsum_p = np.cumsum(sorted_p, axis=1)
        # index = index[0]
        # sorted_logits = sorted_logits[0]
        # cumsum_p = cumsum_p[0] 
        top_p_num = (cumsum_p < top_p).sum(axis=1) + 1

        # Get the corresponding probs and indices
        probs = sorted_logits
        for i, top_p in enumerate(top_p_num):
            probs[i][top_p:] = 0
        # probs = sorted_logits[:top_p_num]
        p_args = index
        # p_args = index[:top_p_num]
        p = probs / probs.sum(axis=1).reshape(-1, 1)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k_num)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k_num)
        # probs = probs[0]
        # p_args = p_args[0]
        # Avoid rounding error
        # if sum(probs) == 0:
        #     probs = np.array([1 / top_k_num for _ in range(top_k_num)])
        p = probs / probs.sum(axis=1).reshape(-1, 1)
    return p, p_args


def generate_increment(model, origin_inputs, config, tokenizer, verbose=False):
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
    # outputs = deepcopy(origin_inputs)
    outputs = [[origin_inputs[i][j] for j in range(valid_length)] for i in range(batch_size)]
    output_codes = ["" for _ in range(batch_size)]
    # If target length exceeds seq_length, use seq_length instead
    target_length = valid_length + max_generate_length
    if verbose:
        print("target_length was ", valid_length, " + ", max_generate_length, " = ", target_length)
    target_length = seq_length if target_length > seq_length else target_length
    if verbose:
        print("target_length is ", target_length)
    gen_end = [False for _ in range(batch_size)]
    allow_comments_next = [True for _ in range(batch_size)]
    allow_comments = [True for _ in range(batch_size)]

    # A list of the frequency of each token
    frequency_list = np.zeros((batch_size, vocab_embedding_vocab_size))
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)),
                       'constant', constant_values=(end_token, end_token))
    if verbose:
        print("input_ids is ", input_ids)

    # Indicate the exact token position
    current_index = valid_length - 1 if valid_length - 1 > 0 else 0
    batch_valid_length = Tensor(np.array([current_index for _ in range(batch_size)]), mstype.int32)
    current_index = Tensor(np.array([current_index + i * seq_length for i in range(batch_size)]), mstype.int32)
    # For first graph, not_init should be false
    init_true = Tensor([True], mstype.bool_)
    init_false = Tensor([False], mstype.bool_)
    init = init_false
    # Claim the first graph
    model.predict_network.add_flags_recursive(is_first_iteration=True)
    # Call a single inference with input size of (bs, seq_length)
    logits = model.predict(Tensor(input_ids, mstype.int32),
                           current_index, init, batch_valid_length)

    # Claim the second graph and set not_init to true
    init = init_true
    model.predict_network.add_flags_recursive(is_first_iteration=False)

    comments_index = [2, ]  # '#': 2, ' #': 1303
    newline_index = [198, ]  # '\n': 198
    # A single loop generates one token, loop until reaching target seq_length or generating eod token
    while valid_length < target_length:
        if all(gen_end):
            break
        # Reshape the output logits
        logits = logits.asnumpy()
        log_probs = logits.reshape(batch_size, vocab_embedding_vocab_size)

        # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
        log_probs_revised = log_probs - frequency_list * \
                            frequency_penalty - (frequency_list > 0) * presence_penalty
        log_probs_revised /= temperature

        bad_words_index = [[] for _ in range(batch_size)]
        # for i in range(batch_size):
        #     if not allow_comments[i]:
        #         bad_words_index[i] += comments_index

        p, p_args = sampler(log_probs_revised, top_p, top_k_num, use_pynative, bad_words_index=bad_words_index)
        # Random select a token as final output for this round
        target_index = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            target_index[i] = np.random.choice(len(p[i]), p=p[i])

        if verbose:
            # print("=== log_probs_revised is", log_probs_revised)
            print("=== p:", p, "shape:", p.shape)
            print("=== p_args:", p_args, "shape", p_args.shape)
            print(
                f"=== Length {valid_length}, target index {target_index}, chosen token {p_args[np.arange(batch_size), target_index]}, generation end status {gen_end}.")

        # Update frequency list
        target = p_args[np.arange(batch_size), target_index]
        frequency_list[np.arange(batch_size), target] = frequency_list[np.arange(batch_size), target] + 1

        batch_valid_length = Tensor(np.array([valid_length for _ in range(batch_size)]), mstype.int32)
        current_index = Tensor(np.arange(batch_size, dtype=np.int32), mstype.int32)
        input_id = Tensor([target], mstype.int32).reshape(-1, 1)
        for i in range(batch_size):
            if not gen_end[i]:
                output_codes[i] += tokenizer.decode_code([int(target[i])])[0]
                if is_code_generation_finished(output_codes[i]):
                    gen_end[i] = True
                    output_codes[i] = truncate_text(output_codes[i])
                if output_codes[i].endswith('#'):
                    allow_comments_next[i] = False
                elif output_codes[i].endswith('\n'):
                    allow_comments[i] = allow_comments_next[i]
                    allow_comments_next[i] = True
            outputs[i].append(int(target[i]))

        # Call a single inference with input size of (bs, 1)
        logits = model.predict(input_id, current_index,
                               init, batch_valid_length)
        valid_length += 1
    return output_codes

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

"""CodeGeeX tokenizers."""

import numpy as np
from abc import ABC
from abc import abstractmethod

from .gpt2_tokenization import GPT2Tokenizer
from transformers import AutoTokenizer


def encode_whitespaces(text: str, start_extra_id: int=10, max_len: int=10):
    """Encode whitespaces to extra tokens.

    >>> encode_whitespaces('a\\n  b\\n   c', 10, 10)
    'a\\n<|extratoken_10|>b\\n<|extratoken_11|>c'
    """
    for i in np.arange(max_len, 1, -1):
        text = text.replace(" " * i, f"<|extratoken_{start_extra_id + i - 2}|>")
    return text


def decode_whitespaces(text: str, start_extra_id: int=10, max_len: int=10):
    """Decode the whitespace-encoded strings produced by encode_whitespace.

    >>> text = 'a\\n  b\\n   c'
    >>> s, l = 10, 10
    >>> text == decode_whitespaces(encode_whitespaces(text, s, l), s, l)
    True
    """
    for l in range(2, max_len + 1):
        token_id = start_extra_id - 2 + l
        token = f"<|extratoken_{token_id}|>"
        text = text.replace(token, " " * l)
    return text


def build_hgf_tokenizer(args):
    """Initialize tokenizer."""
    tokenizer_path = args.tokenizer_path
    if args.rank == 0:
        print(f"> building huggingface tokenizer from {tokenizer_path} ...", flush=True)
    assert tokenizer_path is not None, "Tokenizer path must be provided."

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if args.rank == 0:
        print(f"  > eos_token = {tokenizer.eos_token}", flush=True)

    ws_start_id = args.ws_encoding_start_id if "ws_encoding_start_id" in args else None
    ws_len = args.ws_encoding_length if "ws_encoding_length" in args else None

    return HgfTokenizerWrapper(
        tokenizer, ws_start=ws_start_id, ws_len=ws_len
    )


def build_tokenizer(args):
    """Initialize tokenizer."""
    if "tokenizer_path" in args and args.tokenizer_path is not None:
        # build huggingface tokenizer
        tokenizer = build_hgf_tokenizer(args)
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)
        return tokenizer

    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.
    assert args.vocab_file is not None
    if args.tokenizer_type == "GPT2BPETokenizer":
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    if args.make_vocab_size_divisible_by > orig_vocab_size:
        multiple = args.make_vocab_size_divisible_by
    else:
        multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = "GPT2 BPE"
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(
            vocab_file, merge_file, errors="replace", special_tokens=[], max_len=None
        )
        self.eod_id = self.tokenizer.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class HgfTokenizerWrapper(AbstractTokenizer):
    """Wrapper for Hugging Face tokenizer."""

    def __init__(
            self,
            tokenizer,
            ws_start: int = None,
            ws_len: int = None,
    ):
        super(HgfTokenizerWrapper, self).__init__(tokenizer.__class__.__name__)
        self.tokenizer = tokenizer
        self.ws_start = ws_start
        self.ws_len = ws_len

    def tokenize(self, text):
        if self.ws_start:
            text = encode_whitespaces(text, self.ws_start, self.ws_len)
        input_ids = self.tokenizer(text, is_split_into_words=False).input_ids

        return input_ids

    def detokenize(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if self.ws_start:
            text = decode_whitespaces(text, self.ws_start, self.ws_len)
        return text

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def inv_vocab(self):
        return len(self.tokenizer.decoder)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

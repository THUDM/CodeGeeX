import numpy as np
from typing import *
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2TokenizerFast


def encode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """ Encode whitespaces to extra tokens.

    >>> encode_whitespaces('a\\n  b\\n   c', 10, 10)
    'a\\n<|extratoken_10|>b\\n<|extratoken_11|>c'
    """
    for i in np.arange(max_len, 1, -1):
        text = text.replace(" " * i, f"<|extratoken_{start_extra_id + i - 2}|>")
    return text


def decode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """ Decode the whitespace-encoded strings produced by encode_whitespace.

    >>> text = 'a\\n  b\\n   c'
    >>> s, l = 10, 10
    >>> text == decode_whitespaces(encode_whitespaces(text, s, l), s, l)
    True
    """
    for l in range(2, max_len + 1):
        token_id = start_extra_id - 2 + l
        token = f'<|extratoken_{token_id}|>'
        text = text.replace(token, ' ' * l)
    return text

    
class CodeGeeXTokenizer(object):
    def __init__(
        self, 
        tokenizer: GPT2TokenizerFast = None, 
        tokenizer_path: str = "EleutherAI/gpt-j-6B", 
        start_extra_id: int = 10,
        max_len : int = 10,
        mode='codegeex-13b',
        dict_file: str = None,
    ):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(tokenizer_path)
        if mode not in ['codegeex-13b']:
            raise ValueError(f"Invalid mode {mode}, choose from ['codegeex-13b']")
        self.start_extra_id = start_extra_id
        self.max_len = max_len
        self.mode = mode
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def encode_code(self, code: str):
        if self.mode == 'codegeex-13b':
            code = encode_whitespaces(code, self.start_extra_id, self.max_len)
            input_ids = self.tokenizer(code, is_split_into_words=False, verbose=False).input_ids
            
        return input_ids
    
    def decode_code(self, input_ids):
        if self.mode == 'codegeex-13b':
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False, verbose=False)
            output_code = decode_whitespaces(text, self.start_extra_id, self.max_len)
        
        return output_code
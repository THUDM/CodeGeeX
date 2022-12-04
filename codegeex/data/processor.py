from typing import *
from time import perf_counter

from codegeex.data.data_utils import sliding_window
from codegeex.data.types import PromptSample, LabelSample


class PromptDatasetProcessor(object):
    def __init__(
        self,
        tokenize: Callable,
        pad_token: int,
        keep_order: bool = False,
        max_seq_len: int = 2048,
        sliding_stride: int = 200,
        discard_overlong: bool = True,
        eod_token: int = None, 
        preprocess: Callable = None,
    ):
        super(PromptDatasetProcessor, self).__init__()
        self._keep_order = keep_order
        self._max_seq_len = max_seq_len
        self._sliding_stride = sliding_stride
        self._tokenize = tokenize
        self._pad_token = pad_token
        self._discard_overlong = discard_overlong
        self._eod_token = eod_token
        self._preprocess = preprocess

        self.doc_processed = 0
        self.doc_generated = 0
        self.start_time = 0

    def pad_seq(self, prompt_tokens: List[int], code_tokens: List[int], extra: dict = None) -> Dict[str, List[int]]:
        total_length = len(prompt_tokens) + len(code_tokens)
        assert total_length <= self._max_seq_len, f"padding sequence: {total_length} > {self._max_seq_len}"
        pad_len = self._max_seq_len - total_length
        input_ids = prompt_tokens + code_tokens + [self._pad_token] * pad_len
        attention_mask = [1] * len(prompt_tokens) + [1] * len(code_tokens) + [0] * pad_len
        labels = [-100] * len(prompt_tokens) + code_tokens + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_sample(self, sample: PromptSample) -> Iterable[Dict[str, List[int]]]:
        """
        Process a sample.
        """
        prompt_tokens = self._tokenize(sample.prompt)
        code_tokens = self._tokenize(sample.code)

        if self._eod_token is not None:
            code_tokens.append(self._eod_token)

        if len(prompt_tokens) + len(code_tokens) > self._max_seq_len:
            if self._discard_overlong:
                return
            for p, t in sliding_window(prompt_tokens, code_tokens, self._max_seq_len, self._sliding_stride, self._sliding_stride):
                yield self.pad_seq(p, t)
        else:
            yield self.pad_seq(prompt_tokens, code_tokens, extra=sample.extra)

    def process_sample_strict(self, sample: PromptSample) -> List[Dict[str, List[int]]]:
        """
        Instead of processing lazily, we turn the iterable into a list.
        """
        return list(self.process_sample(sample))

    def process_sample_(self, sample) -> List[Dict[str, List[int]]]:
        prompt_sample = self._preprocess(sample)
        return self.process_sample_strict(prompt_sample)

    def report(self):
        duration = perf_counter() - self.start_time
        process_speed = self.doc_processed * 1.0 / duration
        gen_speed = self.doc_generated * 1.0 / duration
        print(f">>> processed: {self.doc_processed} in {duration:.2f}s, speed: {process_speed:.2f} docs/s")
        print(f"... generated: {self.doc_generated} in {duration:.2f}s, speed: {gen_speed:.2f} docs/s")



class LabelDatasetProcessor(object):
    def __init__(
        self,
        tokenize: Callable,
        pad_token: int,
        keep_order: bool = False,
        max_seq_len: int = 2048,
        sliding_stride: int = 200,
        discard_overlong: bool = True,
        eod_token: int = None, 
        preprocess: Callable = None,
    ):
        super(LabelDatasetProcessor, self).__init__()
        self._keep_order = keep_order
        self._max_seq_len = max_seq_len
        self._sliding_stride = sliding_stride
        self._tokenize = tokenize
        self._pad_token = pad_token
        self._discard_overlong = discard_overlong
        self._eod_token = eod_token
        self._preprocess = preprocess

        self.doc_processed = 0
        self.doc_generated = 0
        self.start_time = 0

    def pad_seq(self, prompt_tokens: List[int], label: int, extra: dict = None) -> Dict[str, List[int]]:
        total_length = len(prompt_tokens) 
        assert total_length <= self._max_seq_len, f"padding sequence: {total_length} > {self._max_seq_len}"
        pad_len = self._max_seq_len - total_length
        input_ids = prompt_tokens +  [self._pad_token] * pad_len
        attention_mask = [1] * len(prompt_tokens) + [0] * pad_len
        label = [label]

        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "length": [len(prompt_tokens)],
                "labels": label
        }
    def process_sample(self, sample: LabelSample) -> Iterable[Dict[str, List[int]]]:
        """
        Process a sample.
        """
        prompt_tokens = self._tokenize(sample.prompt)
        label = sample.label

        
        if len(prompt_tokens) > self._max_seq_len:
            if self._discard_overlong:
                return
            prompt_tokens=prompt_tokens[-self._max_seq_len:]
        
        yield self.pad_seq(prompt_tokens, label, extra=sample.extra)

    def process_sample_strict(self, sample: LabelSample) -> List[Dict[str, List[int]]]:
        """
        Instead of processing lazily, we turn the iterable into a list.
        """
        return list(self.process_sample(sample))

    def process_sample_(self, sample) -> List[Dict[str, List[int]]]:
        prompt_sample = self._preprocess(sample)
        return self.process_sample_strict(prompt_sample)

    def report(self):
        duration = perf_counter() - self.start_time
        process_speed = self.doc_processed * 1.0 / duration
        gen_speed = self.doc_generated * 1.0 / duration
        print(f">>> processed: {self.doc_processed} in {duration:.2f}s, speed: {process_speed:.2f} docs/s")
        print(f"... generated: {self.doc_generated} in {duration:.2f}s, speed: {gen_speed:.2f} docs/s")

from typing import *
from dataclasses import dataclass


@dataclass
class PromptSample:
    prompt: str
    code: str
    extra: dict = None


PromptDataset = Iterable[PromptSample]

@dataclass
class LabelSample:
    prompt: str
    label: int
    extra: dict = None

LabelDataset = Iterable[LabelSample]
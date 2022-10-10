from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torchtyping import TensorType


@dataclass
class PromptElement:
    text: str
    tokens: TensorType["num_tokens"]


@dataclass
class PromptBatch:
    text: Iterable[str]
    tokens: TensorType["batch_size", "num_tokens"]


@dataclass
class AccelerateRLElement:
    output_tokens: TensorType["output_size"]
    rewards: TensorType["output_size"]


@dataclass
class AccelerateRLBatchElement:
    output_tokens: TensorType["batch_size", "output_size"]
    rewards: TensorType["batch_size", "output_size"]

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torchtyping import TensorType


@dataclass
class PPORLElement:
    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    logprobs: TensorType["response_size", "vocab_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size", "vocab_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]

from dataclasses import dataclass
from typing import Callable, Iterable

from torchtyping import TensorType

TensorType


@dataclass
class ILQLElement:
    input_ids: TensorType["query_size"]
    attention_mask: TensorType["query_size"]
    rewards: TensorType["reward_size"]


@dataclass
class ILQLBatch:
    input_ids: TensorType["batch_size", "query_size"]
    attention_mask: TensorType["batch_size", "query_size"]
    rewards: TensorType["batch_size", "reward_size"]

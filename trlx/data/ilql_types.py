from dataclasses import dataclass
from typing import Callable, Iterable

from torchtyping import TensorType

TensorType


@dataclass
class ILQLElement:
    """
    Data element for ILQL

    :param input_ids: Input tokens. Should be a long tensor.
    :type input_ids: torch.Tensor

    :param attention_mask: Attention mask. Should be a long tensor.
    :type attention_mask: torch.Tensor

    :param rewards: Rewards for each token. Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    input_ids: TensorType["query_size"]
    attention_mask: TensorType["query_size"]
    rewards: TensorType["reward_size"]


@dataclass
class ILQLBatch:
    """
    Batched ILQL data elements

    :param input_ids: Batch of input tokens.
    :type input_ids: torch.Tensor

    :param attention_mask: Batch of attention masks.
    :type attention_mask: torch.Tensor

    :param rewards: Batch of rewards for each token in each token batch.
    :type rewards: torch.Tensor
    """

    input_ids: TensorType["batch_size", "query_size"]
    attention_mask: TensorType["batch_size", "query_size"]
    rewards: TensorType["batch_size", "reward_size"]

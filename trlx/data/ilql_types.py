from dataclasses import dataclass, fields

from torchtyping import TensorType  # type: ignore


def flatten_dataclass(cls: type):
    """Return a function that flattens a dataclass into a list"""
    cls_fields = [f.name for f in fields(cls)]
    return lambda x: [getattr(x, f) for f in cls_fields]


def unflatten_dataclass(cls: type):
    """Return a function that unflattens a list into a dataclass"""
    cls_fields = [f.name for f in fields(cls)]
    return lambda x: cls(**dict(zip(cls_fields, x)))


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
    states_ixs: TensorType["states_size"]
    actions_ixs: TensorType["reward_size"]
    dones: TensorType["states_size"]


@dataclass
class ILQLSeq2SeqElement:
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
    decoder_input_ids: TensorType["reward_size"]
    rewards: TensorType["reward_size"]
    states_ixs: TensorType["states_size"]
    actions_ixs: TensorType["reward_size"]
    dones: TensorType["states_size"]


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
    states_ixs: TensorType["batch_size", "states_size"]
    actions_ixs: TensorType["batch_size", "reward_size"]
    dones: TensorType["batch_size", "states_size"]


@dataclass
class ILQLSeq2SeqBatch:
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
    decoder_input_ids: TensorType["batch_size", "reward_size"]
    rewards: TensorType["batch_size", "reward_size"]
    states_ixs: TensorType["batch_size", "states_size"]
    actions_ixs: TensorType["batch_size", "reward_size"]
    dones: TensorType["batch_size", "states_size"]

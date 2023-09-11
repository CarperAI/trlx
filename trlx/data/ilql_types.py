from dataclasses import dataclass
from torchtyping import TensorType  # type: ignore

@dataclass
class ILQLElement:
    """
    A single data item for ILQL training

    :param input_ids: Long tensor of input tokens.
    :type input_ids: torch.Tensor

    :param attention_mask: Attention mask for input tokens. Should be a long tensor.
    :type attention_mask: torch.Tensor

    :param rewards: Rewards for each input token.
    :type rewards: torch.Tensor

    :param states_ixs: Indices of states (user input or environment input for example) in the `input_ids`.
    :type states_ixs: torch.Tensor

    :param actions_ixs: Indices of actions (model output) in the `input_ids` tensor.
    :type actions_ixs: torch.Tensor

    :param dones: Indicator of for the terminal state (end of episode) in the `input_ids` tensor.
    :type dones: torch.Tensor
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
    A single data item for ILQL training

    :param input_ids: Long tensor of input tokens.
    :type input_ids: torch.Tensor

    :param attention_mask: Attention mask for input tokens. Should be a long tensor.
    :type attention_mask: torch.Tensor

    :param decoder_input_ids: Long tensor of target input tokens.
    :type decoder_input_ids: torch.Tensor

    :param rewards: Rewards for each input token.
    :type rewards: torch.Tensor

    :param states_ixs: Indices of states (user input or environment input for example) in the `input_ids`.
    :type states_ixs: torch.Tensor

    :param actions_ixs: Indices of actions (model output) in the `input_ids` tensor.
    :type actions_ixs: torch.Tensor

    :param dones: Indicator of for the terminal state (end of episode) in the `input_ids` tensor.
    :type dones: torch.Tensor
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

    :param states_ixs: Batch of indices of states (user input or environment input for example) in the `input_ids`.
    :type states_ixs: torch.Tensor

    :param actions_ixs: Batch of indices of actions (model output) in the `input_ids` tensor.
    :type actions_ixs: torch.Tensor

    :param dones: Batch of indicators of for the terminal state (end of episode) in the `input_ids` tensor.
    :type dones: torch.Tensor
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

    :param decoder_input_ids: Batch of target input tokens.
    :type decoder_input_ids: torch.Tensor

    :param rewards: Batch of rewards for each token in each token batch.
    :type rewards: torch.Tensor

    :param states_ixs: Batch of indices of states (user input or environment input for example) in the `input_ids`.
    :type states_ixs: torch.Tensor

    :param actions_ixs: Batch of indices of actions (model output) in the `input_ids` tensor.
    :type actions_ixs: torch.Tensor

    :param dones: Batch of indicators of for the terminal state (end of episode) in the `input_ids` tensor.
    :type dones: torch.Tensor
    """

    input_ids: TensorType["batch_size", "query_size"]
    attention_mask: TensorType["batch_size", "query_size"]
    decoder_input_ids: TensorType["batch_size", "reward_size"]
    rewards: TensorType["batch_size", "reward_size"]
    states_ixs: TensorType["batch_size", "states_size"]
    actions_ixs: TensorType["batch_size", "reward_size"]
    dones: TensorType["batch_size", "states_size"]

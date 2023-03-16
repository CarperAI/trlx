from dataclasses import dataclass

from torchtyping import TensorType


@dataclass
class MRTRLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over all tokens in the vocabulary for
                    each token generated from the policy network
                    (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens,
                    with a dimension across the vocabulary.
    :type logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    query_tensor: TensorType["num_candidates", "query_size"]
    response_tensor: TensorType["num_candidates", "response_size"]
    logprobs: TensorType["num_candidates", "response_size", "vocab_size"]
    values: TensorType["num_candidates", "response_size"]
    rewards: TensorType["num_candidates", "response_size"]


@dataclass
class MRTRLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    query_tensors: TensorType["batch_size", "num_candidates", "query_size"]
    response_tensors: TensorType["batch_size", "num_candidates", "response_size"]
    logprobs: TensorType["batch_size", "num_candidates", "response_size", "vocab_size"]
    values: TensorType["batch_size", "num_candidates", "response_size"]
    rewards: TensorType["batch_size", "num_candidates", "response_size"]

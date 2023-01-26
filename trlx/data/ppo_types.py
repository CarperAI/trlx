from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from torchtyping import TensorType
import torch


@dataclass
class PPORLElement:
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

    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    logprobs: TensorType["response_size", "vocab_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
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

    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size", "vocab_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]


RewardFnInput = Union[List[List[str]], Tuple[List[str], List[str], List[str]], List[str]]


@dataclass
class RunElementBatch:
    # TODO have a non-batch version and base this off of that
    query_tensors: List[torch.Tensor]
    padded_samples: List[torch.Tensor]
    logprobs: List[torch.Tensor]
    values: List[torch.Tensor]
    kl_divergence_estimate: List[torch.Tensor]
    str_samples: List[str]
    str_prompts: List[str]
    str_outputs: List[str]

    # Make it so that it can be accessed as a dict
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    # Make it so that two RunElements can be added together.
    # Assume all List attributes are the same length, and add them elementwise
    # Assume the tensors can be added together
    def __add__(self, other : RunElementBatch):
        return RunElementBatch(
            query_tensors=self.query_tensors + other.query_tensors,
            padded_samples=self.padded_samples + other.padded_samples,
            logprobs=self.logprobs + other.logprobs,
            values=self.values + other.values,
            kl_divergence_estimate=self.kl_divergence_estimate + other.kl_divergence_estimate,
            str_samples=self.str_samples + other.str_samples,
            str_prompts=self.str_prompts + other.str_prompts,
            str_outputs=self.str_outputs + other.str_outputs,
        )
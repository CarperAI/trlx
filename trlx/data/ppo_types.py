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
    str_samples: List[List[str]]
    str_prompts: List[List[str]]
    str_outputs: List[List[str]]

    # Make it so that it can be accessed as a dict
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __repr__(self):
        """
        First print the dimensions of the data, then print the data itself, with each field on a new line.
        """
        output = f"query_tensors: {self.query_tensors.shape}\n"
        output += f"padded_samples: {self.padded_samples.shape}\n"
        output += f"logprobs: {len(self.logprobs)} x {self.logprobs[0].shape}\n"
        output += f"values: {len(self.values)} x {self.values[0].shape}\n"
        output += f"kl_divergence_estimate: {len(self.kl_divergence_estimate)} x {self.kl_divergence_estimate[0].shape}\n"
        output += f"str_samples: {len(self.str_samples)} x {len(self.str_samples[0])}\n"
        output += f"str_prompts: {len(self.str_prompts)} x {len(self.str_prompts[0])}\n"
        output += f"str_outputs: {len(self.str_outputs)} x {len(self.str_outputs[0])}\n"
        output += "\n"

        output += f"query_tensors:\n {self.query_tensors}\n"
        output += f"padded_samples:\n {self.padded_samples}\n"
        output += f"logprobs:\n {self.logprobs}\n"
        output += f"values:\n {self.values}\n"
        output += f"kl_divergence_estimate:\n {self.kl_divergence_estimate}\n"
        output += f"str_samples:\n {self.str_samples}\n"
        output += f"str_prompts:\n {self.str_prompts}\n"
        output += f"str_outputs:\n {self.str_outputs}\n"

        return output


        


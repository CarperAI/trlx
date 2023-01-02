import functools
from typing import MutableMapping, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers


def make_head(n_embd: int, out: int) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out),
    )


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_causal_hidden_layers(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


# HuggingFace utilities


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_causal_base_model(model: transformers.AutoModelForCausalLM) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox")
    return findattr(model, decoder_attrs)


def hf_get_causal_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified model.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_causal_hidden_layers(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: transformers.AutoModelForCausalLM) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_num_hidden_layers(config: transformers.PretrainedConfig) -> int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
    """
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer")
    return findattr(config, num_hidden_layers_attrs)


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_from_logits(logits, labels):
    """Compute log softmax values from logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()

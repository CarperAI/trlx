# Copyright 2022 CarperAI & The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file contains a modified version of the `PreTrainedModelWrapper` class from
# HuggingFace's `trl` library. The original source code can be found here:
# https://github.com/lvwerra/trl/blob/78c13226bf8ea1ccd9b1c091f03a938098521f6c/trl/models/modeling_base.py

import functools
import inspect
import json
import os
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download

try:
    from opendelta import (
        AdapterModel,
        BitFitModel,
        LoraModel,
        PrefixModel,
        SoftPromptModel,
    )

    HAS_OPENDELTA = True
except ModuleNotFoundError:
    HAS_OPENDELTA = False


class PreTrainedModelWrapper(nn.Module, transformers.utils.PushToHubMixin):
    """A wrapper around `transformers.PreTrainedModel`

    Reference: @younesbelkada's `PreTrainedModelWrapper`
    https://github.com/lvwerra/trl/blob/4f5c16fafde42d9aca971952bcdcc1f5a0a68cf0/trl/models/modeling_base.py#L2

    Attributes:
        _auto_model_parent_class (transformers.AutoModel): The `transformers.AutoModel`
            type to base the wrapping behavior off of, e.g. `transformers.AutoModelForCausalLM`.
        _supported_modules (List[str]): A list of attribute names for modules of
            the underlying architecture model. This is used, for example, to save
            and load any additional modules by manipulating the state dict.
        _supported_args (List[str]): A list of arguments specific to the underlying
            architecture to separate from arguments that are supported by the
            parent `AutoModel` class. Any arguments that are not supported by the
            underlying model will be passed to the parent `AutoModel` class.
    """

    _auto_model_parent_class: transformers.AutoModel = None
    _supported_modules: List[str] = None
    # TODO (jon-tow): Supported args should come from a `Config` of the specific
    # underlying type similar to how config classes are used to instantiate
    # `transformers.PreTrainedModel`s.
    _supported_args: List[str] = None

    def __init__(
        self, pretrained_model: Optional[transformers.PreTrainedModel] = None, **kwargs
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        # cache `pre_trained.forward` args for general use (avoids incompatible args across architectures)
        self.forward_kwargs = inspect.getfullargspec(self.pretrained_model.forward).args

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separates the kwargs from the arguments that with support inside
        `supported_args` and the ones that we don't.
        """
        supported_kwargs = {}
        unsupported_kwargs = {}
        for key, value in kwargs.items():
            if key in cls._supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value
        return supported_kwargs, unsupported_kwargs

    @classmethod
    def from_config(cls, config):
        """Instantiates the underlying base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.AutoModel.from_pretrained` to load
            the model weights
        """
        pretrained_model = cls._auto_model_parent_class.from_pretrained(config)
        return pretrained_model

    @classmethod
    def from_pretrained(  # noqa: max-complexity
        cls,
        pretrained_model_name_or_path: Union[str, transformers.PreTrainedModel],
        *model_args,
        **kwargs,
    ):
        """
        Instantiate a pretrained pytorch model from a pretrained model configuration.
        This method is a wrapper around `transformers.PreTrainedModel.from_pretrained`.
        Pleas refer to the documentation of `transformers.PreTrainedModel.from_pretrained`
        for more information.

        Args:
            pretrained_model_name_or_path (str or `transformers.PreTrainedModel`):
                The identifier of the pretrained model to load or the pretrained model itself.
            *model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the `_auto_model_parent_class`.
            **kwargs (dict, *optional*):
                Dictionary of keyword arguments to pass to both the underlying `_auto_model_parent_class`
                call (e.g. `transformers.AutoModelForCausalLM.from_pretrained`) and the specific
                instance of the wrapped model.

        NOTE: You must pass in arguments specific to the wrapped model as keyword arguments.
        """
        if kwargs is not None:
            wrapped_model_kwargs, from_pretrained_kwargs = cls._split_kwargs(kwargs)
        else:
            from_pretrained_kwargs = {}
            wrapped_model_kwargs = {}

        if isinstance(pretrained_model_name_or_path, str):
            # Load the pretrained model using the `transformers` AutoClass (e.g. AutoModelForCausalLM)
            pretrained_model = cls._auto_model_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **from_pretrained_kwargs
            )
        elif isinstance(pretrained_model_name_or_path, transformers.PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                f"Invalid type for `pretrained_model_name_or_path`: {type(pretrained_model_name_or_path)}"
                "Expected `str` or `transformers.PreTrainedModel`."
            )

        model = cls(pretrained_model, **wrapped_model_kwargs)

        if isinstance(pretrained_model_name_or_path, str):
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            sharded_index_filename = os.path.join(
                pretrained_model_name_or_path, "pytorch_model.bin.index.json"
            )
            is_sharded = False

            if not os.path.exists(filename):
                try:
                    filename = hf_hub_download(
                        pretrained_model_name_or_path, "pytorch_model.bin"
                    )
                # sharded
                except Exception:
                    if os.path.exists(sharded_index_filename):
                        index_file_name = sharded_index_filename
                    else:
                        index_file_name = hf_hub_download(
                            pretrained_model_name_or_path,
                            "pytorch_model.bin.index.json",
                        )
                    with open(index_file_name, "r") as f:
                        index = json.load(f)
                    # check filename with supported modules
                    files_to_download = set()
                    for k, v in index["weight_map"].items():
                        if any([module in k for module in cls._supported_modules]):
                            files_to_download.add(v)
                    is_sharded = True

            if is_sharded:
                # download each file and add it to the state_dict
                state_dict = {}
                for shard_file in files_to_download:
                    filename = os.path.join(pretrained_model_name_or_path, shard_file)
                    # download if shard file doesn't exist locally
                    if not os.path.exists(filename):
                        filename = hf_hub_download(
                            pretrained_model_name_or_path, shard_file
                        )
                    state_dict.update(torch.load(filename, map_location="cpu"))
            else:
                state_dict = torch.load(filename, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.post_init(state_dict=state_dict)
        return model

    def save_pretrained(self, *args, **kwargs):
        """Save the pretrained model to a directory. This method is a wrapper around
        `transformers.PreTrainedModel.save_pretrained`. Please refer to the documentation
        of `transformers.PreTrainedModel.save_pretrained` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return the state_dict of the pretrained model."""
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        """Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of `pretrained_model.transformer.forward`"""
        # FIXME: This is a hack to get around the fact that the `transformers` architectures
        # we use don't have an enforced consistent API for `forward` parameters.
        return {k: v for k, v in kwargs.items() if k in self.forward_kwargs}


def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def freeze_bottom_seq2seq_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    if num_layers_unfrozen == -1:
        return
    shared_embed = model.shared
    decoder_embed = model.decoder.embed_tokens
    encoder_blocks = model.encoder.block
    encoder_norm_layer = model.encoder.final_layer_norm
    decoder_norm_layer = model.decoder.final_layer_norm
    decoder_blocks = model.decoder.block[:-num_layers_unfrozen]
    blocks_to_freeze = (
        list(encoder_blocks)
        + list(decoder_blocks)
        + [shared_embed]
        + [encoder_norm_layer]
        + [decoder_norm_layer]
        + [decoder_embed]
    )
    for block in blocks_to_freeze:
        block.requires_grad_(False)


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


def hf_get_decoder(model: nn.Module) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox", "decoder")
    return findattr(model, decoder_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
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


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
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


# OpenDelta utilities


MODIFIED_MODULES_DICT = {
    "gptj": {
        "attention": ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        "mlp": ["mlp.fc_in", "mlp.fc_out"],
        "all": [
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.out_proj",
            "mlp.fc_in",
            "mlp.fc_out",
        ],
    },
    "gpt_neox": {
        "attention": ["attention.query_key_value"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "all": [
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    },
    "opt": {
        "attention": [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ],
        "mlp": ["fc1", "fc2"],
        "all": [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ],
    },
    "bloom": {
        "attention": ["self_attention.query_key_value", "self_attention.dense"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "all": [
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    },
    "t5": {
        "attention": [
            "layer.0.SelfAttention.q",
            "layer.0.SelfAttention.k",
            "layer.0.SelfAttention.v",
            "layer.0.SelfAttention.o",
            "layer.1.EncDecAttention.q",
            "layer.1.EncDecAttention.k",
            "layer.1.EncDecAttention.v",
            "layer.1.EncDecAttention.o",
        ],
        "mlp": [
            "layer.2.DenseReluDense.wo",
            "layer.2.DenseReluDense.wi_0",
            "layer.2.DenseReluDense.wi_1",
        ],
        "all": [
            "layer.0.SelfAttention.q",
            "layer.0.SelfAttention.k",
            "layer.0.SelfAttention.v",
            "layer.0.SelfAttention.o",
            "layer.1.EncDecAttention.q",
            "layer.1.EncDecAttention.k",
            "layer.1.EncDecAttention.v",
            "layer.1.EncDecAttention.o",
            "layer.2.DenseReluDense.wo",
            "layer.2.DenseReluDense.wi_0",
            "layer.2.DenseReluDense.wi_1",
        ],
    },
}


def generate_layer_regex(config: transformers.PretrainedConfig, num_layers_unfrozen: int = -1) -> str:
    """Generates a regex range for the specified number of learnable layers."""
    if num_layers_unfrozen == -1:
        return "(\d)+."
    num_hidden_layers = hf_get_num_hidden_layers(config)
    start_layer = num_hidden_layers - num_layers_unfrozen
    if start_layer < 0:
        raise Exception("Number of layers unfrozen cannot be greater than number of layers in the model")
    pattern = f"(?:{regex_for_range(start_layer, num_hidden_layers - 1)})."
    return f"{pattern}"


def get_delta_modified_modules(
    config: transformers.PretrainedConfig,
    modified_modules: List[str],
    num_layers_unfrozen: int = -1,
) -> List[str]:
    """Returns a list of module names to be modified for a given delta method with
    the specified number of learnable layers."""
    unfrozen_layers_pattern = generate_layer_regex(config, num_layers_unfrozen)

    # [r] for regex as per https://github.com/thunlp/OpenDelta/blob/main/opendelta/utils/name_based_addressing.py#L20
    regex_prefix = "[r]"
    # TODO (jon-tow): `decoder.block.` is hardcoded to support T5 layer naming.
    decoder_prefix = "decoder.block." if config.is_encoder_decoder else ""
    module_list = [regex_prefix + decoder_prefix + unfrozen_layers_pattern + module for module in modified_modules]
    return module_list


def get_delta_model_class(model_type: str):
    if not HAS_OPENDELTA:
        raise ValueError("OpenDelta package required to train with delta models. https://github.com/thunlp/OpenDelta.")
    delta_models = {
        "bitfit": BitFitModel,
        "adapter": AdapterModel,
        "prefix": PrefixModel,
        "lora": LoraModel,
        "softprompt": SoftPromptModel,
    }
    return delta_models[model_type]


def parse_delta_kwargs(
    config: transformers.PretrainedConfig,
    delta_kwargs: Dict[str, Any],
    num_layers_unfrozen: int = -1,
) -> Tuple[str, Dict[str, Any]]:
    """Parses through delta kwargs to get delta type and proper modified modules."""
    # This function is needed to parse through the `delta_kwargs` in order to:
    # 1) Get the `delta_type` method name to access the correct `delta_model_class`
    # 2a) Accept user specified `modified_modules` and if not provided use the `trlx` default mapping
    # 2b) Convert the list of `modified_modules` to a range of layers that fit within the range
    #    of learnable layers as specified by `num_layers_unfrozen`

    # Pop `delta_type` to allow passing the kwargs to the model constructor since
    # `delta_type` is not a valid argument of the constructor
    delta_type = delta_kwargs.pop("delta_type")
    assert delta_type in ["lora"], "Only `LoRA` based delta models are supported"

    # Use `trlx` default modified modules if none are specified
    modified_modules = delta_kwargs.get("modified_modules", "all")
    if modified_modules in ["all", "attention", "mlp"]:
        if config.model_type not in MODIFIED_MODULES_DICT:
            raise ValueError(
                f"Model type `{config.model_type}` is not currently supported for "
                "delta training with default modified modules."
            )
        modified_modules = MODIFIED_MODULES_DICT[config.model_type][modified_modules]
    # Update the `modified_modules` with the correct layer ranges
    delta_kwargs["modified_modules"] = get_delta_modified_modules(
        config, modified_modules, num_layers_unfrozen=num_layers_unfrozen
    )

    return delta_type, delta_kwargs


def regex_for_range(min_: int, max_: int) -> str:  # noqa
    """Returns a regex that matches all numbers in the given range.

    Example: regex_for_range(12, 34) -> "1[2-9]|2\d|3[0-4]"

    Copyright (c) 2013, Dmitry Voronin. All rights reserved.
    Reference: https://github.com/voronind/range-regex
    """

    def split_to_patterns(min_, max_):
        subpatterns = []
        start = min_
        for stop in split_to_ranges(min_, max_):
            subpatterns.append(range_to_pattern(start, stop))
            start = stop + 1
        return subpatterns

    def split_to_ranges(min_, max_):
        stops = {max_}
        nines_count = 1
        stop = fill_by_nines(min_, nines_count)
        while min_ <= stop < max_:
            stops.add(stop)
            nines_count += 1
            stop = fill_by_nines(min_, nines_count)
        zeros_count = 1
        stop = fill_by_zeros(max_ + 1, zeros_count) - 1
        while min_ < stop <= max_:
            stops.add(stop)
            zeros_count += 1
            stop = fill_by_zeros(max_ + 1, zeros_count) - 1
        stops = list(stops)
        stops.sort()
        return stops

    def fill_by_nines(integer, nines_count):
        return int(str(integer)[:-nines_count] + "9" * nines_count)

    def fill_by_zeros(integer, zeros_count):
        return integer - integer % 10**zeros_count

    def range_to_pattern(start, stop):
        pattern = ""
        any_digit_count = 0
        for start_digit, stop_digit in zip(str(start), str(stop)):
            if start_digit == stop_digit:
                pattern += start_digit
            elif start_digit != "0" or stop_digit != "9":
                pattern += "[{}-{}]".format(start_digit, stop_digit)
            else:
                any_digit_count += 1
        if any_digit_count:
            pattern += r"\d"
        if any_digit_count > 1:
            pattern += "{{{}}}".format(any_digit_count)
        return pattern

    positive_subpatterns = []
    negative_subpatterns = []

    if min_ < 0:
        min__ = 1
        if max_ < 0:
            min__ = abs(max_)
        max__ = abs(min_)
        negative_subpatterns = split_to_patterns(min__, max__)
        min_ = 0
    if max_ >= 0:
        positive_subpatterns = split_to_patterns(min_, max_)

    negative_only_subpatterns = ["-" + val for val in negative_subpatterns if val not in positive_subpatterns]
    positive_only_subpatterns = [val for val in positive_subpatterns if val not in negative_subpatterns]
    intersected_subpatterns = ["-?" + val for val in negative_subpatterns if val in positive_subpatterns]
    subpatterns = negative_only_subpatterns + intersected_subpatterns + positive_only_subpatterns
    return "|".join(subpatterns)

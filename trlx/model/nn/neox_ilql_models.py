import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, fields
from itertools import chain
from typing import Union, Sequence, Any, TypeVar, Tuple

from torchtyping import TensorType  # type: ignore

from trlx.model.nn.ilql_models import ILQLConfig, ILQLHeads
from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import register_method, MethodConfig


import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

import wandb

import megatron
from megatron import print_rank_0, mpu
from megatron.neox_arguments import NeoXArgs
from megatron.utils import get_ltor_masks_and_position_ids
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec  # type: ignore


class HeadsLayerSpec(LayerSpec):
    def __init__(self, specs: Sequence[LayerSpec]):
        super().__init__(Heads)
        self.branches = specs

    def build(self):
        return Heads(heads=[m.build() for m in self.branches])


class Heads(nn.Module):
    def __init__(self, heads: Sequence[nn.Module]):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor):
        return [m(x) for m in self.heads]


from megatron.model import GPT2ModelPipe
from megatron.model.utils import Lambda


def get_layer_name(layer):
    if isinstance(layer, LayerSpec):
        return layer.typename.__name__
    elif isinstance(layer, nn.Module):
        return layer.__class__.__name__
    else:
        return layer.__name__


def lift_to_layerspec(x):
    if isinstance(x, LayerSpec):
        return x
    elif isinstance(x, nn.Module):
        return LayerSpec(lambda: x)
    else:  # assuming function
        return LayerSpec(Lambda, x)


def get_callable(x):
    if isinstance(x, LayerSpec):
        return x.build()
    elif isinstance(x, nn.Module):
        return x
    else:  # assuming function
        return x


class ParMapLayerSpec(LayerSpec):
    def __init__(self, specs):
        class NamedFlatmap(FlatMap):
            pass

        NamedFlatmap.__name__ = ",".join(get_layer_name(spec) for spec in specs)

        super().__init__(NamedFlatmap)
        self.specs = specs

    def build(self):
        # print([(type(m), get_layer_name(m)) for m in self.specs])
        return self.typename(modules=[get_callable(m) for m in self.specs])


class FlatMap(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.branches = nn.ModuleList(modules)

    def forward(self, xs: Sequence[Any]):
        return type(xs)(
            filter(
                lambda x: x is None, (branch(x) for x, branch in zip(xs, self.branches))
            )
        )


class PassRLData(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = get_callable(model)

    def forward(self, args):
        # DeepSpeed is hardcoded with attention_mask last
        (states_ixs, actions_ixs), model_args = args[:2], args[2:]

        if len(model_args) == 1:
            m = self.model(model_args[0])
        else:
            m = self.model(model_args)

        if not isinstance(m, (tuple, list)):
            m = (m,)

        return (states_ixs, actions_ixs, *m)


class DropRLData(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = get_callable(model)

    def forward(self, args):
        # DeepSpeed is hardcoded with attention_mask last
        (states_ixs, actions_ixs), model_args = args[:2], args[2:]
        if len(model_args) == 1:
            m = self.model(model_args[0])
        else:
            m = self.model(model_args)

        return m


class CallWithRLData(nn.Module):
    def __init__(self, model: LayerSpec):
        super().__init__()
        self.model = get_callable(model)

    def forward(self, args):
        (states_ixs, actions_ixs), model_args = args[:2], args[2:]
        return self.model(
            *model_args, states_ixs=states_ixs.long(), actions_ixs=actions_ixs.long()
        )


def get_named_subclass(cls, name):
    class Sub(cls):
        pass

    Sub.__name__ = name

    return Sub


class Drop(nn.Identity):
    def forward(self, x):
        return None


T = TypeVar("T")


def broadcast_dataclass(obj: T) -> T:
    d = {
        k: mpu.broadcast_data([k], obj.__dict__, v.dtype)[k]
        for k, v in obj.__dict__.items()
    }
    return type(obj)(**d)


DTYPE_MAP = {
    "input_ids": torch.long,
    "attention_mask": torch.long,
    "rewards": torch.float,
    "states_ixs": torch.long,
    "actions_ixs": torch.long,
    "dones": torch.long,
}


def preprocess_batch(neox_args: NeoXArgs, b: ILQLBatch):

    field_types = [(field.name, field.type) for field in fields(ILQLBatch)]
    field_types = sorted(field_types, key=lambda nt: nt[0])

    # on non-zero ranks, the input batch is none
    data_dict = None if b is None else b.__dict__
    data_dict = {
        k: mpu.broadcast_data([k], data_dict, DTYPE_MAP[k])[k]
        for k, dtype in field_types
    }

    b = ILQLBatch(**data_dict)

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=b.input_ids,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
    )

    return (b.input_ids, position_ids, attention_mask,), (
        b.input_ids,
        b.attention_mask,
        b.rewards,
        b.states_ixs,
        b.actions_ixs,
        b.dones,
    )
    return (b.states_ixs, b.actions_ixs, b.input_ids, position_ids, attention_mask,), (
        b.input_ids,
        b.attention_mask,
        b.rewards,
        b.states_ixs,
        b.actions_ixs,
        b.dones,
    )


class GPTNeoXWithValueHeads(GPT2ModelPipe):
    def __init__(
        self,
        config: megatron.NeoXArgs,
        ilql_config: ILQLConfig,
    ):
        super().__init__(
            neox_args=config,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
            use_cache=False,
        )

        def wrap_loss(outputs, labels):
            """Unflatten labels from tuples to ILQLBatch"""
            input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones = labels
            labels = ILQLBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                rewards=rewards,
                states_ixs=states_ixs,
                actions_ixs=actions_ixs,
                dones=dones,
            )

            logits, (qs, target_qs, vs) = outputs
            qs = [
                q.gather(1, actions_ixs.unsqueeze(-1).repeat(1, 1, q.shape[-1]))
                for q in qs
            ]
            target_qs = [
                q.gather(1, actions_ixs.unsqueeze(-1).repeat(1, 1, q.shape[-1]))
                for q in target_qs
            ]
            vs = vs.gather(1, states_ixs.unsqueeze(-1).repeat(1, 1, vs.shape[-1]))
            outputs = (logits, (qs, target_qs, vs))

            return ilql_config.loss(outputs, labels)[0]

        self.loss_fn = wrap_loss

        embedding = self.specs[-1]
        new_specs = [
            LayerSpec(get_named_subclass(PassRLData, get_layer_name(ls)), ls)
            for ls in self.specs[:-1]
        ] + [
            HeadsLayerSpec(
                specs=[
                    LayerSpec(DropRLData, embedding),
                    LayerSpec(
                        CallWithRLData,
                        LayerSpec(
                            ILQLHeads,
                            ilql_config,
                            self.hidden_size,
                            config.padded_vocab_size,
                        ),
                    ),
                ]
            )
        ]
        new_specs = self.specs[:-1] + [
            HeadsLayerSpec(
                specs=[
                    embedding,
                    LayerSpec(
                        ILQLHeads,
                        ilql_config,
                        self.hidden_size,
                        config.padded_vocab_size,
                    ),
                ]
            )
        ]

        self.specs = new_specs

        PipelineModule.__init__(
            self,
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=config.pipe_partition_method,
            checkpointable_layers=["GMLPBlock", "ParallelTransformerLayerPipe"],
        )

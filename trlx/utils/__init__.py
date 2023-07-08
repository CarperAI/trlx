import importlib
import math
import os
import random
import subprocess
import time
from dataclasses import is_dataclass
from enum import Enum
from itertools import repeat
from numbers import Number
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


def is_peft_available():
    return importlib.util.find_spec("peft") is not None


def print_rank_0(*message):
    """
    Print only once from the main rank
    """
    if os.environ.get("RANK", "0") == "0":
        print(*message)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


def set_seed(seed: int):
    """
    Sets seeds across package dependencies for reproducibility.
    """
    seed += int(os.environ.get("RANK", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Training utils


def get_distributed_config(accelerator: Accelerator):
    """
    Return accelerator distributed config
    """

    dist_config = {
        "mixed_precision": accelerator.mixed_precision,
        "num_gpus": accelerator.num_processes,
    }

    if accelerator.state.deepspeed_plugin is not None:
        ds_plugin = accelerator.state.deepspeed_plugin
        dist_config.update(
            {
                "gradient_accumulation_steps": ds_plugin.gradient_accumulation_steps,
                "gradient_clipping": ds_plugin.gradient_clipping,
                "zero_stage": ds_plugin.zero_stage,
                "offload_optimizer_device": ds_plugin.offload_optimizer_device,
                "offload_param_device": ds_plugin.offload_param_device,
            }
        )

    return dist_config


class OptimizerName(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    ADAM_8BIT_BNB: str = "adam_8bit_bnb"
    ADAMW_8BIT_BNB: str = "adamw_8bit_bnb"
    SGD: str = "sgd"


def get_optimizer_class(name: OptimizerName):
    """
    Returns the optimizer class with the given name

    Args:
        name (str): Name of the optimizer as found in `OptimizerNames`
    """
    if name == OptimizerName.ADAM:
        return torch.optim.Adam
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW
    if name == OptimizerName.ADAM_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import Adam8bit

            return Adam8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use the 8-bit Adam. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.ADAMW_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import AdamW8bit

            return AdamW8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use 8-bit AdamW. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerName]
    raise ValueError(f"`{name}` is not a supported optimizer. " f"Supported optimizers are: {supported_optimizers}")


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    if name == SchedulerName.LINEAR:
        return LinearLR
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(f"`{name}` is not a supported scheduler. " f"Supported schedulers are: {supported_schedulers}")


class Clock:
    """
    Helper object for keeping track of time for computations.
    """

    def __init__(self):
        self.start = time.time()
        self.total_time = 0
        self.total_samples = 0

    def tick(self, samples: int = 0) -> float:
        """
        Returns time (s) since last call to tick(). Also records samples processed since last call.

        :param samples: number of samples that have been processed since last call
        """
        end = time.time()
        delta = end - self.start
        self.start = end

        if samples != 0:
            self.total_time += delta
            self.total_samples += samples

        return delta

    def get_stat(self, n_samp: int = 1000, reset: bool = False):
        """
        Returns average time (s) per n_samp samples processed

        :param reset: Reset counts?
        """
        sec_per_samp = self.total_time / self.total_samples

        if reset:
            self.total_samples = 0
            self.total_time = 0

        return sec_per_samp * n_samp


def tree_map(f, tree: Any) -> Any:
    """
    Apply function f to all leaves in tree
    """
    if is_dataclass(tree):
        return tree.__class__(**{k: tree_map(f, v) for k, v in tree.__dict__.items()})
    elif isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return tree.__class__(tree_map(f, v) for v in tree)
    else:
        return f(tree)


def to_device(tree, device, non_blocking=False):
    """
    Move all tensors in tree to device
    """
    return tree_map(lambda x: x.to(device, non_blocking=non_blocking), tree)


def filter_non_scalars(xs: Dict) -> Dict:
    """
    Trims everything that can't be casted to float
    """
    ys = {}
    for k, v in xs.items():
        try:
            ys[k] = float(v)
        except TypeError:
            continue

    return ys


def get_git_tag() -> Tuple[str, str]:
    """
    Returns commit's short hash and date
    """
    try:
        output = subprocess.check_output("git log --format='%h/%as' -n1".split())
        branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
        return branch.decode()[:-1], output.decode()[1:-2]
    except subprocess.CalledProcessError:
        return "unknown", "unknown"


# Iter utils


def infinite_dataloader(dataloader: Iterable, sampler=None) -> Iterable:
    """
    Returns a cyclic infinite dataloader from a finite dataloader
    """
    epoch = 0
    for _ in repeat(dataloader):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        yield from dataloader

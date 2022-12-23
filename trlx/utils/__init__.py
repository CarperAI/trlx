import os
import random
import time
from enum import Enum
from functools import reduce
from typing import Any, Iterable, List, Dict
from dataclasses import is_dataclass
import subprocess

import numpy as np
import torch
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR
from torchtyping import TensorType

import accelerate
from accelerate import Accelerator


def set_seed(seed: int):
    """
    Sets seeds across package dependencies for reproducibility.
    """
    seed += int(os.environ.get("RANK", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def flatten(L: Iterable[Iterable[Any]]) -> Iterable[Any]:
    """
    Flatten a list of lists into a single list (i.e. [[1, 2], [3, 4]] -> [1,2,3,4])
    """
    return list(reduce(lambda acc, x: acc + x, L, []))


def chunk(L: Iterable[Any], chunk_size: int) -> List[Iterable[Any]]:
    """
    Chunk iterable into list of iterables of given chunk size
    """
    return [L[i : i + chunk_size] for i in range(0, len(L), chunk_size)]


# Training utils


def rampup_decay(ramp_steps, decay_steps, decay_target, opt):
    return ChainedScheduler(
        [
            LinearLR(opt, decay_target, 1, total_iters=ramp_steps),
            LinearLR(opt, 1, decay_target, total_iters=decay_steps),
        ]
    )


def safe_mkdir(path: str):
    """
    Make directory if it doesn't exist, otherwise do nothing
    """
    if os.path.isdir(path):
        return
    os.mkdir(path)


def get_distributed_config(accelerator: Accelerator):
    """
    Return accelerator distributed config
    """

    accelerate_config = accelerator.state
    dist_config = {
        "mixed_precision": accelerate_config.mixed_precision,
        "num_gpus": accelerate_config.num_processes,
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


class OptimizerNames(Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    SGD: str = "sgd"


def get_optimizer_class(name: str):
    """
    Returns the optimizer class with the given name
    """
    if name == OptimizerNames.ADAM.value:
        return torch.optim.Adam
    if name == OptimizerNames.ADAMW.value:
        return torch.optim.AdamW
    if name == OptimizerNames.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerNames]
    raise ValueError(
        f"`{name}` is not a supported optimizer. "
        f"Supported optimizers are: {supported_optimizers}"
    )


class SchedulerNames(Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING: str = "cosine_annealing"


def get_scheduler_class(name: str):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerNames.COSINE_ANNEALING.value:
        return torch.optim.lr_scheduler.CosineAnnealingLR
    supported_schedulers = [s.value for s in SchedulerNames]
    raise ValueError(
        f"`{name}` is not a supported scheduler. "
        f"Supported schedulers are: {supported_schedulers}"
    )


# Stats


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


# Sampling


def topk_mask(xs: TensorType["Batch", "Vocab"], k: int):
    """
    Takes batched distribution over tokens and masks out scores for tokens
    that are not in the top k for that distribution.
    """

    # Get topk per distribution
    # For each dist, getting last value gives k-th largest
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs), xs)


# Sentiment/scores


def sentiment_score(sentiments: Iterable[float]):
    """
    Return tensor of scores in [-1, 1] from sentiment analysis pipeline output
    """
    sentiments = torch.tensor(
        [-s["score"] if s["label"] == "NEGATIVE" else s["score"] for s in sentiments]
    )
    return sentiments


def tree_map(f, tree):
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


def to_device(tree, device):
    """
    Move all tensors in tree to device
    """
    return tree_map(lambda x: x.to(device), tree)


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


def get_git_tag() -> str:
    """
    Returns commit's short hash and date
    """
    output = subprocess.check_output("git log --format='%h/%as' -n1".split())
    branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
    return f"{branch.decode()[:-1]}/{output.decode()[1:-2]}"

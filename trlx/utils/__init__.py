import os
import time
from functools import reduce
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR
from torchtyping import TensorType


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


def add_stat(stats, name, xs, mask, n):
    mean = (xs * mask).sum() / n
    stats.update(
        {
            f"{name}/mean": mean,
            f"{name}/min": torch.where(mask.bool(), xs, np.inf).min(),
            f"{name}/max": torch.where(mask.bool(), xs, -np.inf).max(),
            f"{name}/std": torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
        }
    )


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and std of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs):
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            all_xs = [torch.empty_like(xs) for _ in range(dist.get_world_size())]
            dist.all_gather(all_xs, xs)
            xs = torch.stack(all_xs)

        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        m_a = self.var * self.count
        m_b = xs_var * xs_count
        m_2 = m_a + m_b + delta**2 * self.count * xs_count / tot_count

        self.mean += delta * xs_count / tot_count
        self.var = m_2 / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()

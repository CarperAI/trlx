from functools import reduce

from typing import Iterable, List, Any, Callable

def flatten(L : Iterable[Iterable[Any]]) -> Iterable[Any]:
    """
    Flatten a list of lists into a single list (i.e. [[1, 2], [3, 4]] -> [1,2,3,4])
    """
    return list(reduce(lambda acc, x: acc + x, L, []))

def chunk(L : Iterable[Any], chunk_size : int) -> List[Iterable[Any]]:
    """
    Chunk iterable into list of iterables of given chunk size
    """
    return [L[i:i+chunk_size] for i in range(0, len(L), chunk_size)]

# Training utils

from torch.optim.lr_scheduler import LinearLR, ChainedScheduler

def rampup_decay(ramp_steps, decay_steps, decay_target, opt):
    return ChainedScheduler(
        [
            LinearLR(opt, decay_target, 1, total_iters = ramp_steps),
            LinearLR(opt, 1, decay_target, total_iters= decay_steps)
        ]
    )
    
import os

def safe_mkdir(path : str):
    """
    Make directory if it doesn't exist, otherwise do nothing
    """
    if os.path.isdir(path):
        return
    os.mkdir(path)

import time

class Clock:
    """
    Helper object for keeping track of time for computations.
    """
    def __init__(self):
        self.start = time.time()
        self.total_time = 0
        self.total_samples = 0

    def tick(self, samples : int = 0) -> float:
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

    def get_stat(self, n_samp : int = 1000, reset : bool = False):
        """
        Returns average time (s) per n_samp samples processed

        :param reset: Reset counts?
        """
        sec_per_samp = self.total_time / self.total_samples

        if reset:
            self.total_samples = 0
            self.total_time = 0
            
        return sec_per_samp * n_samp
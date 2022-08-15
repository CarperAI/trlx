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
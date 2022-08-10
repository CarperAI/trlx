from dataclasses import dataclass

from typing import Any, List, Iterable
from torchtyping import TensorType

import random

@dataclass
class GeneralElement:
    pass

@dataclass
class SimElement:
    """
    Batch element for Gyarados or Gyarados-like similarity scoring model
    """
    content : Any = None
    preference : Any = None
    score : float = None

@dataclass
class RLElement:
    state : Any = None
    action : Any = None
    reward : float = None

@dataclass
class BatchElement:
    """
    General batch element for any transformer to use in its forward pass
    """
    tokens : TensorType["BATCH", "SEQ_LEN"]
    masks : TensorType["BATCH", "SEQ_LEN"]

class RolloutStore:
    def __init__(self, capacity = -1):
        self.history : Iterable[RLElement] = []
        self.capacity = capacity

    def get_size(self):
        return len(self.history)
    
    def sample(self, size : int) -> List[RLElement]:
        return random.sample(self.history, size)
    
    def push(self, exps : Iterable[RLElement]):
        for experience in exps:
            self.history.append(experience)
            if len(self.history) > self.capacity:
                del self.history[0]

    
from dataclasses import dataclass

from typing import Any, List, Iterable, Callable
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
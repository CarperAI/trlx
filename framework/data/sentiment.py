from dataclasses import dataclass

from typing import Iterable, Callable
from torchtyping import TensorType

import torch

@dataclass
class SentimentGeneralElement:
    text : Iterable[str]

@dataclass
class SentimentRLElement:
    text : Iterable[str]
    score : TensorType["N"] # Corresponds to sentiment
    action : TensorType["N"] = None
from dataclasses import dataclass

from typing import Iterable, Callable
from torchtyping import TensorType

import torch

def cat_sentiment_general_element(elem1, elem2):
    return SentimentGeneralElement(elem1.txt + elem2.txt)

def cat_sentiment_rl_element(elem1, elem2):
    return SentimentRLElement(
        elem1.txt + elem2.txt,
        torch.cat((elem1.score, elem2.score)),
        None if elem1.action is None else torch.cat((elem1.action, elem2.action))
    )

@dataclass
class SentimentGeneralElement:
    text : Iterable[str]
    cat : Callable = cat_sentiment_general_element

@dataclass
class SentimentRLElement:
    text : Iterable[str]
    score : TensorType["N"]
    action : TensorType["N"] = None
    cat : Callable = cat_sentiment_rl_element

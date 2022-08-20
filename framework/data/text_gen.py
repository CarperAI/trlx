from dataclasses import dataclass

from typing import List, Callable, Iterable

from framework.data import SimElement, RLElement, GeneralElement

@dataclass
class TextGeneralElement(GeneralElement):
    text : str = None

@dataclass
class TextSimBatchElement(SimElement):
    content : Iterable[str] = None
    preference : Iterable[str] = None

@dataclass
class TextRLElement(RLElement):
    state : str = None
    action : int = None
    reward : float = None
    
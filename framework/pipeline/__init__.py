from torch.utils.data import Dataset
from datasets import load_from_disk
import random

from typing import Iterable, Any
from abc import abstractmethod

from framework.data import GeneralElement, RLElement

class BasePipeline(Dataset):
    def __init__(self, path : str = "dataset"):
        pass

    def __getitem__(self, index : int) -> GeneralElement:
        pass

    def __len__(self) -> int:
        pass

class BaseRolloutStore(Dataset):
    def __init__(self, capacity = -1):
        self.history : Iterable[Any] = None
        self.capacity = capacity
    
    @abstractmethod
    def push(self, exps):
        pass

    def __getitem__(self, index : int) -> RLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)


    
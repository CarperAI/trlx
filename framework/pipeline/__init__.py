from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import random

from typing import Iterable, Any, Dict, Callable
from abc import abstractmethod, abstractstaticmethod
import sys

from framework.data import GeneralElement, RLElement

# specifies a dictionary of architectures
_DATAPIPELINE: Dict[str, any] = {}  # registry


def register_datapipeline(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

@register_datapipeline
class BasePipeline(Dataset):
    def __init__(self, path : str = "dataset"):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index : int) -> GeneralElement:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def create_loader(self, batch_size : int, shuffle : bool) -> DataLoader:
        """
        Create a dataloader for the pipeline
        """
        pass



class BaseRolloutStore(Dataset):
    def __init__(self, capacity = -1):
        self.history : Iterable[Any] = None
        self.capacity = capacity
    
    @abstractmethod
    def push(self, exps : Iterable[Any]):
        """
        Push experiences to rollout storage
        """
        pass

    def __getitem__(self, index : int) -> RLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    @abstractmethod
    def create_loader(self, batch_size : int, shuffle : bool, prep_fn : Callable = None) -> DataLoader:
        """
        Create a dataloader for the rollout store

        :param prep_fn: Generic placeholder for some preprocessing function that might need to get passed in (typically a tokenizer)
        :type prep_fn: Callable
        """
        pass    
import random
import sys
from abc import abstractmethod, abstractstaticmethod
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Iterable

from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import BatchEncoding

from trlx.data import GeneralElement, RLElement
from trlx.utils import logging

# specifies a dictionary of architectures
_DATAPIPELINE: Dict[str, any] = {}  # registry

logger = logging.get_logger(__name__)


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
    def __init__(self, path: str = "dataset"):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index: int) -> GeneralElement:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a dataloader for the pipeline

        :param prep_fn: Typically a tokenizer. Applied to GeneralElement after collation.
        """
        pass


class BaseRolloutStore(Dataset):
    def __init__(self, capacity=-1):
        self.history: Iterable[Any] = None
        self.capacity = capacity

    @abstractmethod
    def push(self, exps: Iterable[Any]):
        """
        Push experiences to rollout storage
        """
        pass

    def __getitem__(self, index: int) -> RLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    @abstractmethod
    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a dataloader for the rollout store

        :param prep_fn: Applied to RLElement after collation (typically tokenizer)
        :type prep_fn: Callable
        """
        pass


class MiniBatchIterator:
    """
    A custom iterator for generating mini-batches from a PyTorch DataLoader.
    """

    def __init__(self, data_loader, mb_size, num_mb):
        """
        Initializes the MiniBatchIterator.

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to generate mini-batches from.
            mb_size (int): The size of each mini-batch.
            num_mb (int): The number of mini-batches to generate for each iteration.
        """
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.mb_size = mb_size
        self.num_mb = num_mb

    def __iter__(self):
        return self

    def __next__(self):  # noqa: C901
        batch = next(self.data_loader_iter)
        if batch is None:
            logger.warning(
                "WARNING: Not enough samples to saturate the minibatch size. Increase the number "
                "of prompts or samples or decrease the minibatch size."
            )
            raise StopIteration

        minibatches = []

        for mbi in range(self.num_mb):
            sliced_data = {}
            batch_dict = batch
            if is_dataclass(batch):
                batch_dict = batch.__dict__
            for key, value in batch_dict.items():
                start_idx = mbi * self.mb_size
                end_idx = (mbi + 1) * self.mb_size
                sliced_data[key] = value[start_idx:end_idx]

                if self.num_mb > 1 and len(sliced_data[key]) == 0:
                    logger.warning(
                        "WARNING: MiniBatchIterator generated a minibatch with 0 elements. "
                        "This may be due to the wrong mb_size and/or num_mb or the last batch"
                        "in the dataset being smaller."
                    )
                    sliced_data.pop(key)
                    break
                elif self.num_mb > 1 and len(sliced_data[key]) < self.mb_size:
                    logger.warning(
                        "WARNING: MiniBatchIterator generated a minibatch with fewer elements than mb_size. "
                        "This may be due to the wrong mb_size and/or num_mb or the last batch in the dataset "
                        "being smaller."
                    )
            if not sliced_data:
                break

            if isinstance(batch, BatchEncoding):
                minibatch = BatchEncoding(sliced_data)
            elif is_dataclass(batch):
                minibatch = batch.__class__(**sliced_data)
            # else:
            #     minibatch = sliced_data

            minibatches.append(minibatch)

        if not minibatches:
            raise StopIteration

        return minibatches

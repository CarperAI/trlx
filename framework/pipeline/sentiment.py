from datasets import load_dataset
from functools import partial, reduce
from typing import Tuple, Iterable, Callable
from torchtyping import TensorType

import torch
from torch.utils.data import DataLoader

from framework.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
from framework.data.sentiment import SentimentGeneralElement, SentimentRLElement

@register_datapipeline
class SentimentPipeline(BasePipeline):
    def __init__(self, texts=None):
        super().__init__()
        self.texts = texts

    def __getitem__(self, index : int):
        return self.texts[index]

    def __len__(self) -> int:
        return len(self.texts)

    def create_loader(self, batch_size : int, shuffle : bool, prep_fn : Callable = None) -> DataLoader:
        # def collate_fn(elems : Iterable[SentimentGeneralElement]) -> SentimentGeneralElement:
        #     # We assume actions are unset as elems are being read from rollout storage
        #     return [elem.text for elem in elems]

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class SentimentRolloutStorage(BaseRolloutStore):
    def __init__(self):
        super().__init__()

        self.history : Iterable[Iterable[str], TensorType["N"]] = [[], torch.empty(0)]

    def push(self, exps : Tuple[Iterable[str], TensorType["N"]]):
        txt, r = exps
        self.history[0] += txt
        self.history[1] = torch.cat([self.history[1], r])

    def __getitem__(self, index : int) -> SentimentRLElement:
        return SentimentRLElement(
            self.history[0][index],
            self.history[1][index]
        )

    def __len__(self) -> int:
        return len(self.history[0])

    def create_loader(self, batch_size : int, prep_fn : Callable = None):
        """
        Create dataloader for the sentiment task.

        :param prep_fn: Should be tokenizing function
        :type prep_fn: Callable[Iterable[str], Dict[str, torch.tensor]]
        """
        def collate_fn(elems : Iterable[SentimentRLElement]):
            res = SentimentRLElement(
                    [elem.text for elem in elems],
                    torch.tensor([elem.score for elem in elems])
            )
            if prep_fn is not None:
                return prep_fn(res)
            else:
                return res

        return DataLoader(self, batch_size, shuffle=True, collate_fn = collate_fn)

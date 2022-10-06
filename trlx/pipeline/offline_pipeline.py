from functools import partial, reduce
from typing import Callable, Iterable, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtyping import TensorType

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@register_datapipeline
class OfflinePipeline(BasePipeline):
    def __init__(self, texts=None):
        super().__init__()
        self.texts = texts

    def __getitem__(self, index: int):
        return self.texts[index]

    def __len__(self) -> int:
        return len(self.texts)

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class OfflineRolloutStorage(BaseRolloutStore):
    def __init__(self, input_ids, attention_mask, rewards):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards

    def __getitem__(self, index: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[index], self.attention_mask[index], self.rewards[index]
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int, eos_token_id=0):
        def collate_fn(elems: Iterable[ILQLElement]):
            return ILQLBatch(
                pad_sequence(
                    [x.input_ids for x in elems],
                    batch_first=True,
                    padding_value=eos_token_id,
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards for x in elems], batch_first=True, padding_value=0
                ),
            )

        return DataLoader(
            self, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

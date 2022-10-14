from copy import deepcopy
from functools import partial
from typing import Callable, Iterable, Tuple

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtyping import TensorType
from transformers import DataCollatorWithPadding

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@register_datapipeline
class OfflinePipeline(BasePipeline):
    def __init__(self, prompts, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = list(map(tokenizer if tokenizer else (lambda x: x), prompts))

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int) -> DataLoader:
        collate_fn = (
            DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        )
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)


class OfflineRolloutStorage(BaseRolloutStore):
    def __init__(self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix], self.attention_mask[ix], self.rewards[ix], self.states_ixs[ix], self.actions_ixs[ix], self.dones[ix]
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        def collate_fn(elems: Iterable[ILQLElement]):
            return ILQLBatch(
                pad_sequence(
                    [x.input_ids for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards for x in elems], batch_first=True, padding_value=0.0
                ),
                pad_sequence(
                    [x.states_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.actions_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.dones for x in elems], batch_first=True, padding_value=0
                ),
            )

        return DataLoader(
            self, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

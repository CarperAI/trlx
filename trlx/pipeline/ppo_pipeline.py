import json
import os
import time
from functools import partial
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore


def ppo_collate_fn(padding_side: str, pad_token_id: int, elems: Iterable[PPORLElement]):
    if padding_side == "left":
        # Left padding of already left-padded queries
        query_tensors = pad_sequence(
            [elem.query_tensor.flip(0) for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        ).flip(1)
    elif padding_side == "right":
        query_tensors = pad_sequence(
            [elem.query_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        )
    else:
        raise ValueError(f"Invalid padding side: {padding_side}")

    return PPORLBatch(
        query_tensors,
        # Right pad the rest, to have a single horizontal query/response split
        pad_sequence(
            [elem.response_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        ),
        pad_sequence(
            [elem.logprobs for elem in elems],
            padding_value=0.0,
            batch_first=True,
        ),
        pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
        pad_sequence(
            [elem.rewards for elem in elems],
            padding_value=0.0,
            batch_first=True,
        ),
    )


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str, only_text=True):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        def filter_text(d, only_text):
            if only_text:
                keys = list(d.keys())
                for key in keys:
                    if key != "query_tensor" and key != "response_tensor":
                        d.pop(key)
            return d

        data = [filter_text(exp_to_dict(exp), only_text) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            self, batch_size, shuffle=shuffle, collate_fn=partial(ppo_collate_fn, self.padding_side, self.pad_token_id)
        )

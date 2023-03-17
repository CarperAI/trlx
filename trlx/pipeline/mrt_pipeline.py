import json
import os
import time
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.mrt_types import MRTRLBatch, MRTRLElement
from trlx.pipeline import BaseRolloutStore


class MRTRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training MRT
    """

    def __init__(self, pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[MRTRLElement] = [None]

    def push(self, exps: Iterable[MRTRLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> MRTRLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[MRTRLElement]):
            return MRTRLBatch( # TODO: make sure this is expected
                pad_sequence(
                    [elem.query_tensor.transpose(0, 1) for elem in elems],
                    padding_value=self.pad_token_id,
                ).transpose(0, 1).transpose(1, 2),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor.transpose(0,1) for elem in elems],
                    padding_value=self.pad_token_id,
                ).transpose(0, 1).transpose(1, 2),
                pad_sequence(
                    [elem.logprobs.transpose(0, 1) for elem in elems],
                    padding_value=0.0,
                ).transpose(0, 1).transpose(1, 2),
                pad_sequence(
                    [elem.values.transpose(0, 1) for elem in elems],
                    padding_value=0.0
                ).transpose(0, 1).transpose(1, 2),
                pad_sequence(
                    [elem.rewards.transpose(0, 1) for elem in elems],
                    padding_value=0.0
                ).transpose(0, 1).transpose(1, 2)
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)

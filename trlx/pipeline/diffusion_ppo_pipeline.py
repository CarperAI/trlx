from typing import Iterable

import torch
from torch.utils.data import DataLoader

from trlx.data.diffusion_types import DiffPPORLElement, DiffPPORLBatch
from trlx.pipeline.ppo_pipeline import PPORolloutStorage

class DiffusionPPORolloutStorage(PPORolloutStorage):
    """
    Rollout storage for training diffusion PPO
    """

    def __init__(self):
        super().__init__(None)

    def create_loader(
        self,
        batch_size,
        shuffle
    ) -> DataLoader:
        def collate_fn(elems : Iterable[DiffPPORLElement]):
            return DiffPPORLBatch(
                prompt=torch.stack([elem.prompt for elem in elems]),
                attention_mask=torch.stack([elem.attention_mask for elem in elems]),
                img=torch.stack([elem.img for elem in elems]),
                logprobs=torch.stack([elem.logprobs for elem in elems]),
                values=torch.stack([elem.values for elem in elems]),
                rewards=torch.stack([elem.rewards for elem in elems]),
            )

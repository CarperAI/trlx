from functools import partial, reduce
from typing import Callable, Iterable, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchtyping import TensorType

from trlx.data.accelerate_base_datatypes import PromptBatch, PromptElement
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
from torch.nn.utils.rnn import pad_sequence


@register_datapipeline
class PPOPipeline(BasePipeline):
    """
    Pipeline for training PPO on the IMDB review dataset. The task is to generate positive reviews.
    """
    def __init__(self, tokenizer, config, prompt_dataset_path = None):
        super().__init__()

        ds = load_dataset("imdb", split="test")
        ds = ds.rename_columns({"text": "review", "label": "sentiment"})
        ds = ds.filter(lambda x: len(x["review"]) < 500, batched=False)

        self.tokens = [
            tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=config.train.input_size,
                return_tensors="pt",
            )["input_ids"]
            .long()
            .flatten()
            for text in ds["review"]
        ]
        self.text = [tokenizer.decode(tokens.tolist()) for tokens in self.tokens]

    def __getitem__(self, index: int) -> PromptElement:
        return PromptElement(self.text[index], self.tokens[index])

    def __len__(self) -> int:
        return len(self.text)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
        prep_fn: Callable = None,
        num_workers: int = 0,
    ) -> DataLoader:
        # TODO(dahoas): Decide how to support varying sizes of prompts without having to tokenize on fly
        def collate_fn(elems: Iterable[PromptElement]) -> PromptElement:
            return PromptBatch(
                [elem.text for elem in elems],
                torch.stack(
                    [elem.tokens for elem in elems]
                ),  # Assumes token tensors all same size
            )

        return DataLoader(
            self, batch_size, shuffle, collate_fn=collate_fn, num_workers=num_workers
        )


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """
    def __init__(self, pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                # Left padding of already left-padded queries
                pad_sequence([elem.query_tensor.flip(0) for elem in elems], padding_value=self.pad_token_id, batch_first=True).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence([elem.response_tensor for elem in elems], padding_value=self.pad_token_id, batch_first=True),
                pad_sequence([elem.logprobs for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence([elem.rewards for elem in elems], padding_value=0.0, batch_first=True),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)

from datasets import load_dataset
from functools import partial, reduce
from typing import Tuple, Iterable, Callable
from torchtyping import TensorType

import torch
from torch.utils.data import DataLoader

from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
from trlx.data.accelerate_base_datatypes import AccelerateRLBatchElement, AccelerateRLElement, PromptBatch, PromptElement

def filter_outliers(x):
    return len(x['review']) > 200

def process_data(dataset):
    dataset = dataset.rename_columns({'text': 'review', 'label': 'sentiment'})
    dataset = dataset.filter(filter_outliers, batched=False)
    dataset = dataset
    return dataset
    

@register_datapipeline
class AcceleratePipeline(BasePipeline):
    def __init__(self, prompt_dataset_path = None):
        super().__init__()

    def __getitem__(self, index : int) -> PromptElement:
        return PromptElement(self.text[index], self.tokens[index])
    
    def __len__(self) -> int:
        return len(self.text)

    def create_loader(self, batch_size : int, shuffle : bool, prep_fn : Callable = None, num_workers : int = 0) -> DataLoader:
        #TODO(dahoas): Decide how to support varying sizes of prompts without having to tokenize on fly
        def collate_fn(elems : Iterable[PromptElement]) -> PromptElement:
            return PromptBatch(
                [elem.text for elem in elems], torch.stack([elem.tokens for elem in elems])  # Assumes token tensors all same size
            )
        
        return DataLoader(self, batch_size, shuffle, collate_fn = collate_fn, num_workers = num_workers)


class AccelerateRolloutStorage(BaseRolloutStore):
    def __init__(self):
        super().__init__()

        self.history : Iterable[Tuple[TensorType["traj_len"], TensorType["traj_len"]]] = [[], torch.empty(0)]
    
    def push(self, exps : Iterable[Tuple[TensorType["traj_len"], TensorType["traj_len"]]]):
        self.history += exps

    def __getitem__(self, index : int) -> AccelerateRLElement:
        return AccelerateRLElement(
            self.history[index][0],
            self.history[index][1]
        )

    def __len__(self) -> int:
        return len(self.history)
    
    def create_loader(self, batch_size : int, shuffle : bool, prep_fn : Callable = None, num_workers : int = 0) -> DataLoader:
        """
        Create dataloader for the sentiment task.

        :param prep_fn: Should be tokenizing function
        :type prep_fn: Callable[Iterable[str], Dict[str, torch.tensor]]
        """
        #TODO(dahoas): Decide how to support varying sizes of prompts without having to tokenize on fly
        def collate_fn(elems : Iterable[AccelerateRLElement]):
            res = AccelerateRLBatch(
                    torch.stack([elem.output_tokens for elem in elems]),
                    torch.stack([elem.rewards for elem in elems])  # Assumes token tensors all same size
            )
            if prep_fn is not None:
                return prep_fn(res)
            else:
                return res
        
        return DataLoader(self, batch_size, shuffle, collate_fn = collate_fn, num_workers = num_workers)

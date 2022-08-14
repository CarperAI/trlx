from datasets import load_dataset
from functools import partial, reduce
from typing import Tuple, Iterable
from torchtyping import TensorType

import torch

from framework.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline
from framework.data.sentiment import SentimentGeneralElement, SentimentRLElement

def filter_outliers(x):
    return len(x['review']) > 200

def process_data(dataset):
    dataset = dataset.rename_columns({'text': 'review', 'label': 'sentiment'})
    dataset = dataset.filter(filter_outliers, batched=False)
    dataset = dataset
    return dataset

@register_datapipeline
class SentimentPipeline(BasePipeline):
    def __init__(self, path = None):
        ds_train = load_dataset('imdb', split='train+test')
        ds_train = process_data(ds_train)
        self.text = ds_train["text"]

    def __get__item(self, index : int) -> SentimentGeneralElement:
        return SentimentGeneralElement(self.text[index])
    
    def __len__(self) -> int:
        return len(self.text)

class SentimentRolloutStorage(BaseRolloutStore):
    def __init__(self):
        super().__init__()

        self.history : Iterable[Iterable[str], TensorType["N"]] = [[], torch.empty(0)]
    
    def push(self, exps : Tuple[Iterable[str], TensorType["N"]]):
        txt, r = exps
        self.history[0] += txt
        self.history[1] = torch.cat((self.history[1], r))

    def __getitem__(self, index : int) -> SentimentRLElement:
        return SentimentRLElement(
            self.history[0][index],
            self.history[1][index]
        )

    def __len__(self) -> int:
        return len(self.history)

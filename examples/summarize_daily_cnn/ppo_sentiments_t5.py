import os
import pathlib
from typing import Dict, List

import yaml
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig
from transformers import AutoTokenizer

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


config_path = pathlib.Path(__file__).parent / "configs/ppo_config_sentiments.yml"
with config_path.open() as f:
    default_config = yaml.safe_load(f)

import numpy as np


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)



def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    
    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments


    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/t5-imdb")

    output_min_length = 16
    output_max_length = 32
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    
    def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    def build_imdb_dataset_test(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="test")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds



    dataset = build_imdb_dataset(tokenizer)
    prompts = dataset["query"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]
    
    
    trlx.train(
        prompts=prompts,
        eval_prompts=val_prompts,
        reward_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()

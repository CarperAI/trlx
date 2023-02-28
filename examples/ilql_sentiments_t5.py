import os
from typing import Dict, List

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

import trlx
from trlx.data.configs import TRLConfig


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


config_path = "configs/ilql_config.yml"
with config_path.open() as f:
    default_config = yaml.safe_load(f)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    tokenizer = AutoTokenizer.from_pretrained("lvwerra/t5-imdb")

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            size = input_size()
            sample["input_ids"] = tokenizer.encode(sample["review"])[:size] + [tokenizer.eos_token_id]
            sample["decoder_input_ids"] = tokenizer.encode(sample["review"])[size:] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            sample["output"] = tokenizer.decode(sample["decoder_input_ids"])
            sample["output_prompt"] = [[sample["query"], sample["output"]]]
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
            size = input_size()
            sample["input_ids"] = tokenizer.encode(sample["review"])[:size] + [tokenizer.eos_token_id]
            sample["decoder_input_ids"] = tokenizer.encode(sample["review"])[size:] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            sample["output"] = tokenizer.decode(sample["decoder_input_ids"])
            sample["output_prompt"] = [[sample["query"], sample["output"]]]
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    dataset = build_imdb_dataset(tokenizer)
    prompts_outputs = sum(dataset["output_prompt"], [])
    rewards = [float(x) for x in dataset["label"]]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:1000]
    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        eval_prompts=val_prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()

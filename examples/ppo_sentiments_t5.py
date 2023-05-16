import json
import os
import sys
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=100,
        total_steps=100000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        save_best=False,
    ),
    model=ModelConfig(
        model_path="lvwerra/t5-imdb",
        num_layers_unfrozen=-1,
        model_arch_type="seq2seq",
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="lvwerra/t5-imdb",
        padding_side="right",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-5,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=12,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 50,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1,
            "eos_token_id": -1,
        },
    ),
)


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
        self.rng = np.random.default_rng(seed=2023)

    def __call__(self):
        return self.rng.choice(self.values)


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

    def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["review"] = sample["review"].replace("/>br", "")
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
            sample["review"] = sample["review"].replace("/>br", "")
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
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

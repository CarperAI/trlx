import os
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
from trlx.models.modeling_ilql import ILQLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=128,
        epochs=100,
        total_steps=1000,
        batch_size=32,
        checkpoint_interval=1000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
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
    method=ILQLConfig(
        name="ILQLConfig",
        tau=0.7,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.001,
        beta=0,
        steps_for_target_q_sync=5,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=56, top_k=20, beta=4, temperature=1.0),
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
        return dict(sentiments=sentiments)

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/t5-imdb")

    def build_imdb_dataset_test(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="test")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["review"] = sample["review"].replace("/>br", "")
            input_ids = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(input_ids)
            return sample

        ds = ds.map(tokenize, batched=False)
        return ds

    dataset = load_dataset("imdb", split="train")
    prompts = dataset["text"]
    rewards = dataset["label"]
    val_prompts = build_imdb_dataset_test(tokenizer)["query"][0:100]

    trlx.train(
        samples=prompts,
        rewards=rewards,
        eval_prompts=val_prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()

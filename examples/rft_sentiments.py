# This script trains a model to output positive reviews
# using rejection finetuning with a sentiment classifier reward function.
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.trainer.accelerate_rft_trainer import RFTConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=100,
        total_steps=1000,
        batch_size=32,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateRFTTrainer",
    ),
    model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
    method=RFTConfig(
        name="RFTConfig",
        n_generations_per_prompt=4,
        start_percentile=0.9,
        end_percentile=0.95,
        n_improve_steps=1,
        gen_kwargs=dict(
            max_new_tokens=40,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
        ),
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train[:512]")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

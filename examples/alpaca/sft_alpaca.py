import json
import os
from argparse import ArgumentParser
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def preprocess(instruction: str, input: str, output: str):
    """Build Alpaca prompt and output from instruction and input/output examples"""
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request."
        )
        prompt = f"{prefix}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        return [prompt, output]
    else:
        prefix = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )
        prompt = f"{prefix}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        return [prompt, output]


def main(hparams={}, model_name="EleutherAI/gpt-j-6B", dataset="tatsu-lab/alpaca"):
    config = default_sft_config()
    config = config.evolve(
        train=dict(
            total_steps=2400,
            batch_size=4,
            seq_length=1024,
        ),
        model=dict(
            model_path=model_name,
        ),
        tokenizer=dict(
            tokenizer_path=model_name,
        ),
        optimizer=dict(kwargs=dict(lr=2e-5)),
        scheduler=dict(kwargs=dict(eta_min=2e-5)),
        method=dict(
            gen_kwargs=dict(
                max_new_tokens=256,
            )
        ),
    )

    # Merge sweep config with default config if given
    config = TRLConfig.update(config.to_dict(), hparams)

    # alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = load_dataset(dataset, split="train")
    alpaca = [preprocess(x["instruction"], x["input"], x["output"]) for x in alpaca]

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(outputs)))
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="test")
    bad_reviews = imdb.filter(lambda sample: sample["label"] == 0).select(range(256))
    zs_rewrite = [preprocess("Rewrite the input into a positive review.", x["text"][:1024], "")[0] for x in bad_reviews]

    trainer = trlx.train(
        samples=alpaca,
        eval_prompts=zs_rewrite,
        metric_fn=metric_fn,
        config=config,
    )

    slug = f"{model_name.split('/')[-1]}-{dataset.split('/')[-1]}"
    trainer.save_pretrained(f"{slug}-sft")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("override_hparams", type=str, default="{}", nargs="?")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")

    args = parser.parse_args()
    hparams = json.loads(args.override_hparams)

    main(hparams, args.model_name, args.dataset)

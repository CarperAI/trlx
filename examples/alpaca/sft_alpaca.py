import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def preprocess(instruction: str, input: str, output: str):
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context."
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


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams)
    config = config.evolve(
        train=dict(
            total_steps=1200,
            batch_size=4,
        ),
        model=dict(
            model_path="EleutherAI/gpt-j-6B",
        ),
        tokenizer=dict(
            tokenizer_path="EleutherAI/gpt-j-6B",
        ),
        optimizer=dict(kwargs=dict(lr=2e-5)),
        scheduler=dict(kwargs=dict(eta_min=2e-5)),
        method=dict(
            gen_kwargs=dict(
                max_new_tokens=256,
            )
        ),
    )
    # gpt2
    # config = config.evolve(
    #     model=dict(
    #         model_path="gpt2",
    #     ),
    #     tokenizer=dict(
    #         tokenizer_path="gpt2",
    #     ),
    # )
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = [preprocess(x["instruction"], x["input"], x["output"]) for x in alpaca]

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        responses = [sample.split("### Response:\n")[1] for sample in samples]
        sentiments = list(map(get_positive_score, sentiment_fn(responses)))
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="test")
    bad_reviews = imdb.filter(lambda sample: sample["label"] == 0).select(range(256))
    zs_rewrite = [preprocess("Rewrite this into a positive review.", x["text"][:1024], "")[0] for x in bad_reviews]

    trainer = trlx.train(
        samples=alpaca,
        eval_prompts=zs_rewrite,
        metric_fn=metric_fn,
        config=config,
    )
    trainer.save_pretrained("alpaca-sft")


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

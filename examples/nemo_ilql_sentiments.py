import os
import re
from typing import Dict, List

import yaml
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = yaml.safe_load(
    open(os.path.dirname(__file__) + "/../configs/nemo_ilql_config.yml")
)


def split_dialog(dialog):
    dialog = re.split(r"(\n\nHuman: |\n\nAssistant: )", dialog)[1:]
    return ["".join(dialog[:-1]), dialog[-1]]


def preprocess(sample):
    sample["prompt_output"] = [
        split_dialog(sample["chosen"]),
        split_dialog(sample["rejected"]),
    ]
    sample["reward"] = [1, -1]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Anthropic/hh-rlhf").map(preprocess)

    # Flatten [postive, negative] samples into a single list
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])

    test_dataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="helpful-base", split="test"
    ).map(preprocess)
    eval_prompts = [sample[0][0] for sample in test_dataset["prompt_output"]]  # [:1024]

    def reward_fn(xs):
        return [1.0 for x in xs]

    trlx.train(
        dataset=(prompts_outputs, rewards),
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda xs: {"rewards": reward_fn(xs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )

    # sentiment_fn = pipeline(
    #     "sentiment-analysis",
    #     "lvwerra/distilbert-imdb",
    #     top_k=2,
    #     truncation=True,
    #     batch_size=256,
    #     device=-1 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    # )

    # def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
    #     sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    #     return {"sentiments": sentiments}

    # imdb = load_dataset("imdb", split="train+test")

    # trlx.train(
    #     dataset=(imdb["text"], imdb["label"]),
    #     eval_prompts=["I don't know much about Hungarian underground"] * 128 * 20,
    #     metric_fn=metric_fn,
    #     config=config,
    # )


if __name__ == "__main__":
    main()

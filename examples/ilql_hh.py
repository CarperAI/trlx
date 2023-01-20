import math
import os
import re

import numpy as np
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.configs import TRLConfig

config_path = os.path.join(os.path.dirname(__file__), "configs/ilql_hh.yml")
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-hh")


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


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

    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"

    dataset = load_dataset("Anthropic/hh-rlhf").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])

    test_dataset = load_dataset(
        "Anthropic/hh-rlhf", data_dir="helpful-base", split="test"
    ).map(preprocess)
    eval_prompts = [sample[0][0] for sample in test_dataset["prompt_output"]][:1024]

    trlx.train(
        dataset=(prompts_outputs, rewards),
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda xs: {"rewards": reward_fn(xs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

import math
import os
import re

import numpy as np
import tritonclient.grpc as client_util
import yaml
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from datasets import load_dataset
from trlx.data.configs import TRLConfig

config_path = os.path.join(os.path.dirname(__file__), "configs/ppo_hh.yml")
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-static")


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def reward_fn(samples, prompts, outputs):
        samples = [s[len(hhh_prompt) :] for s in samples]
        input = reward_tokenizer(samples, padding=True, max_length=1024)

        mbs = 8
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)
            attention_mask = np.array(input.attention_mask[batch_ixs], dtype=np.int8)

            inputs = [
                prepare_tensor("input_ids", input_ids),
                prepare_tensor("attention_mask", attention_mask),
            ]

            result = client.infer(triton_model, inputs)
            rewards = result.as_numpy("rewards")
            if rewards is None:
                raise RuntimeError("No output data")

            last_ixs = attention_mask.sum(-1, keepdims=True) - 1
            returns = np.take_along_axis(rewards, last_ixs, -1)
            out.extend(returns.flatten())

        return out

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    hhh_prompt = ""  # .join(dataset['test'][[-4, -24, -28]]['chosen'])

    dialogues = sum([[x["chosen"], x["rejected"]] for x in dataset["train"]], [])
    dialogues = [re.split(r"(\n\nHuman: |\n\nAssistant: )", x)[1:] for x in dialogues]
    prompts = [hhh_prompt + "".join(x[:-1]) for x in dialogues]
    eval_prompts = [hhh_prompt + "".join(x[:-1]) for x in dialogues][:128]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
        stop_sequences=["Human:", "human:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

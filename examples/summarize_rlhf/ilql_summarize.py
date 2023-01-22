import os
from typing import List
import yaml

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

import numpy as np
import tritonclient.grpc as client_util
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.configs import TRLConfig
import math

config_path = os.path.join(os.path.dirname(__file__), "configs/ilql_summarize.yml")
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-summarize")

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

    def reward_fn(samples, **kwargs):
        samples = [s + reward_tokenizer.eos_token for s in samples]
        input = reward_tokenizer(samples, padding=True, max_length=1024)

        mbs = 24
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

            result = client.infer(triton_model, [
                prepare_tensor("input_ids", input_ids),
            ])

            rewards = result.as_numpy("rewards")
            if rewards is None:
                raise RuntimeError("No output data")

            out.extend(rewards.flatten())

        return out

    def preprocess(sample):
        sample["prompt_output"] = [
            [sample['prompt'] + ' TL;DR:', sample['chosen'][7:]],
            [sample['prompt'] + ' TL;DR:', sample['rejected'][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    dataset = dataset.map(preprocess)

    prompts_outputs = sum(dataset['train']['prompt_output'], [])
    rewards = sum(dataset['train']['reward'], [])

    eval_prompts = [
        sample[0][0]
        for sample in dataset['test']['prompt_output']
    ]
    eval_prompts = list(set(eval_prompts))[:64]

    trainer = trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=lambda samples, **kwargs: {'rewards': reward_fn(samples)},
        eval_prompts=eval_prompts,
        config=config,
    )

if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

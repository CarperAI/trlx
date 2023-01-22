import os
import yaml
import pathlib
from typing import List

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

config_path = os.path.join(os.path.dirname(__file__), "configs/ppo_summarize.yml")
default_config = yaml.safe_load(open(config_path))

triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-summarize")

def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def main(hparams):
    config = TRLConfig.update(default_config, hparams)

    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def reward_fn(samples, prompts, outputs):
        # get rewards for the original samples as well (TODO: precompute)
        samples += [p + original_outputs[p] for p in prompts]
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

        sampled_rewards = torch.tensor(out[:len(prompts)])
        original_rewards = torch.tensor(out[len(prompts):])
        normed_rewards = sampled_rewards - original_rewards

        return normed_rewards

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    def retokenize_prompt(sample):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        prompt = tokenizer.decode(
            tokenizer(
                sample["prompt"].split("TL;DR:")[0],
                truncation=True,
                max_length=max_length
                - 5,  # to make sure "TL;DR" dont get truncated
            )["input_ids"],
            skip_special_tokens=True,
        ).strip() + "\nTL;DR:"
        sample["prompt"] = tokenizer.decode(
            tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        return sample

    dataset = load_dataset("CarperAI/openai_summarize_tldr").map(
        retokenize_prompt, num_proc=os.cpu_count() // 2
    )
    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    original_outputs = {}
    for i in range(len(train_posts)):
        original_outputs[train_posts[i]] = train_summaries[i]
    for i in range(len(val_posts)):
        original_outputs[val_posts[i]] = val_summaries[i]

    trainer = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=train_posts,
        eval_prompts=val_posts[:128],
        config=config,
    )
    trained.save_pretrained("ppo_summarize_ckpt")

if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)


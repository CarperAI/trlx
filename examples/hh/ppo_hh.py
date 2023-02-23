import math
import os

import numpy as np
import torch
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.configs import TRLConfig

config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/ppo_hh.yml"))
default_config = yaml.safe_load(open(config_path))


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_reward_fn():  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    triton_host = os.environ.get("TRITON_HOST")

    if triton_host:
        triton_url, triton_model = triton_host.split("/")
        client = client_util.InferenceServerClient(url=triton_url, verbose=False)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, max_length=1024)

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

                result = client.infer(triton_model, [prepare_tensor("input_ids", input_ids)])
                rewards = result.as_numpy("rewards")
                out.extend(rewards)

            return out

    elif os.environ.get("RANK", "0") == "0":

        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path, eos_token_id):
                super().__init__()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                self.transformer = model.transformer
                self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
                self.eos_token_id = eos_token_id

            def forward(self, input_ids):
                states = self.transformer(input_ids)[0]
                rewards = self.v_head(states).squeeze(-1)
                ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
                returns = torch.gather(rewards, 1, ends).squeeze(-1)
                return returns

        reward_model = RewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
        directory = snapshot_download("Dahoas/gptj-rm-static", revision="676bfd4d")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint))
        reward_model.eval()
        reward_model.requires_grad_(False)
        device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(device)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(
                device
            )

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = reward_model(input_ids)
                out.extend(rewards)

            return out

    else:
        reward_fn = True

    return reward_fn


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/rm-static")
    prompts = dataset["train"]["prompt"]
    eval_prompts = dataset["test"]["prompt"][:280]
    reward_fn = create_reward_fn()

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def create_steamshp_reward_fn(device=None):
    tokenizer = T5Tokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)

    A_token = tokenizer("A")["input_ids"][0]

    def batched_reward_fn(_, prompts, outputs):
        rm_inputs = [
            f"POST: {prompt} \n\n RESPONSE A: {output}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE"
            for prompt, output in zip(prompts, outputs)
        ]

        rm_inputs = tokenizer(rm_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**rm_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        scores = torch.softmax(outputs.scores[0], dim=-1)[:, A_token]
        return scores.cpu().tolist()

    def reward_fn(samples, prompts, outputs):
        mbs = 32
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            rewards = batched_reward_fn(samples[batch_ixs], prompts[batch_ixs], outputs[batch_ixs])
            out.extend(rewards)

        return out

    return reward_fn


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = default_ppo_config()
    config.model.model_path = "EleutherAI/gpt-j-6b"
    config.model.num_layers_unfrozen = 2
    config.tokenizer.tokenizer_path = "gpt2"
    config.train.seq_length = 512
    config.train.batch_size = 4
    config.optimizer.kwargs["lr"] = 8e-6
    config.scheduler.kwargs["eta_min"] = 8e-6
    config.method.num_rollouts = 64
    config.method.chunk_size = 16
    config.train.total_steps = 6000

    # SteamSHP reccomends truncating from the left
    config.tokenizer.truncation_side = "left"

    config = TRLConfig.update(config.to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    reward_fn = create_steamshp_reward_fn(device)

    dataset = load_dataset("Dahoas/rm-static")
    prompts = dataset["train"]["prompt"]
    eval_prompts = dataset["test"]["prompt"][:280]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

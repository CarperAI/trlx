import json
import math
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
    default_nemo_2b_config,
    default_nemo_20b_config,
    default_ppo_config,
)


def create_steamshp_reward_fn(device=None):
    tokenizer = T5Tokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
    tokenizer.truncation_side = "left"
    model = T5ForConditionalGeneration.from_pretrained("stanfordnlp/SteamSHP-flan-t5-large")
    model.requires_grad_(False)
    model.eval()
    model = model.to("cpu")

    A_token = tokenizer("A")["input_ids"][0]

    def batched_reward_fn(_, prompts, outputs):
        rm_inputs = [
            f"POST: {prompt} \n\n RESPONSE A: {output}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE"
            for prompt, output in zip(prompts, outputs)
        ]

        rm_inputs = tokenizer(rm_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**rm_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        scores = torch.softmax(outputs.scores[0], dim=-1)[:, A_token]
        return scores

    def reward_fn(samples, prompts, outputs):
        mbs = 32
        out = []
        model.to(device, non_blocking=True)
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            rewards = batched_reward_fn(samples[batch_ixs], prompts[batch_ixs], outputs[batch_ixs])
            out.append(rewards)
        model.to("cpu", non_blocking=True)
        return torch.cat(out)

    return reward_fn


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    cfg_name = os.environ.get("NEMO_CONFIG", "1.3B")
    if cfg_name == "1.3B":
        nemo_config = default_nemo_1_3b_config()
    elif cfg_name == "2B":
        nemo_config = default_nemo_2b_config()
    elif cfg_name == "20B":
        nemo_config = default_nemo_20b_config()
    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    config = default_config.evolve(
        train=dict(
            total_steps=10000,
            seq_length=2048,
            batch_size=8,
            epochs=100,
            eval_interval=64,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=f"/mnt/hdd/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=256,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_steamshp",
            seed=2023,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "steamshp", cfg_name],
        ),
        tokenizer=dict(
            truncation_side="left",
        ),
        optimizer=dict(
            name="distributed_fused_adam",
            kwargs=dict(
                lr=6e-6,
                weight_decay=1e-06,
                betas=(0.9, 0.95),
            ),
        ),
        scheduler=dict(
            name="CosineAnnealing",
        ),
        model=dict(num_layers_unfrozen=2),
        method=dict(
            num_rollouts=128,
            init_kl_coef=0.044,
            vf_coef=0.94,
            gen_kwargs=dict(temperature=1.0, max_new_tokens=256),
            chunk_size=128,
            ppo_epochs=1,
        ),
    )
    config.scheduler.kwargs = dict(warmup_steps=4, constant_steps=1e12, min_lr=5e-06)

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % 8

    reward_fn = create_steamshp_reward_fn(local_rank)  # Take few words off of movies reviews as prompts
    dataset = load_dataset("Dahoas/rm-static")
    prompts = dataset["train"]["prompt"]
    eval_prompts = dataset["test"]["prompt"][:256]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
        stop_sequences=["Human:", "Assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

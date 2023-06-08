import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
    default_nemo_1_3b_config,
    default_nemo_20b_config,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="NeMoPPOTrainer",
        checkpoint_dir="checkpoints/nemo_ppo_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=128,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


config_name = os.environ.get("CONFIG_NAME")

if config_name == "1.3B":
    nemo_config = default_nemo_1_3b_config()
    default_config.train.trainer_kwargs["megatron_cfg"] = nemo_config
    default_config.train.trainer_kwargs["pretrained_model"] = "/fsx/home-uwu/nemo-megatron-gpt-1.3B/"

    default_config.train.batch_size = 32
    default_config.train.total_steps = 2500
    default_config.optimizer.kwargs["lr"] = 6e-6
    default_config.scheduler.kwargs["eta_min"] = 6e-6
    default_config.train.checkpoint_dir = "checkpoints/nemo_ppo_hh_1.3B"

elif config_name == "20B":
    default_config.train.trainer_kwargs["megatron_cfg"] = default_nemo_20b_config()
    default_config.train.trainer_kwargs["pretrained_model"] = "/fsx/home-uwu/nemo-megatron-gpt-20B/"

    default_config.train.seq_length = 512
    default_config.train.batch_size = 32
    default_config.train.total_steps = 8000
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.train.checkpoint_dir = "checkpoints/nemo_ppo_hh_20B"

    default_config.method.num_rollouts = 16
    default_config.method.chunk_size = 4
    default_config.method.ppo_epochs = 2


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

    return reward_fn


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/rm-static")
    prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]
    reward_fn = create_reward_fn()

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

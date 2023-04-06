import json
import os
import sys
from itertools import islice

from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        batch_size=4,
        epochs=100,
        total_steps=20000,
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="checkpoints/ilql_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1000000000, eta_min=1e-6)),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.6,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.0001,
        beta=0,
        steps_for_target_q_sync=1,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=128, top_k=20, beta=[1, 4], temperature=1.0),
    ),
)

config_name = os.environ.get("CONFIG_NAME")
if config_name == "125M":
    default_config.train.batch_size = 16
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_125M"
    default_config.model.model_path = "EleutherAI/pythia-125m-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
elif config_name == "1B":
    default_config.train.batch_size = 8
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_1B"
    default_config.model.model_path = "EleutherAI/pythia-1.4b-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
elif config_name == "6B":
    default_config.train.batch_size = 4
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_6B"
    default_config.model.model_path = "EleutherAI/pythia-6.9b-deduped"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
elif config_name == "20B":
    default_config.train.batch_size = 1
    default_config.train.total_steps = 3000
    default_config.train.checkpoint_dir = "checkpoints/ilql_hh_20B"
    default_config.model.model_path = "EleutherAI/gpt-neox-20b"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"


def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    sample["reward"] = [1, -1]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])

    rewards = sum(dataset["train"]["reward"], [])
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]
    reward_fn = create_reward_fn()

    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

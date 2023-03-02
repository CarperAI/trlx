import os

import torch
import yaml
from datasets import load_dataset
from omegaconf import OmegaConf
from ppo_hh import create_reward_fn
from transformers import T5ForConditionalGeneration, T5Tokenizer

import trlx
from trlx.data.configs import TRLConfig
from trlx.data.default_configs import default_ilql_config


def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    sample["reward"] = [1, -1]
    return sample


def main(hparams={}):
    config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/ilql_hh.yml"))
    nemo_config_path = os.path.join(os.path.dirname(__file__), "../../configs/nemo_configs/megatron_20b.yml")
    default_config = TRLConfig.from_dict(yaml.safe_load(open(config_path)))

    nemo_cfg = OmegaConf.load(nemo_config_path)
    nemo_cfg.exp_manager.explicit_log_dir = "ilql_hh_logs"
    nemo_cfg.exp_manager.name = "megatron_gpt_20b_ilql_hh_null"
    nemo_cfg.exp_manager.wandb_logger_kwargs.name = "megatron_gpt_20b_ilql_hh_null"

    default_config = default_config.evolve(
        train=dict(
            seq_length=1024,
            batch_size=512,
            total_steps=200,
            trainer="NeMoILQLTrainer",
            trainer_kwargs=dict(
                pretrained_model="/mnt/nvme/home/uwu/nemo-megatron-gpt-20B/",
                megatron_cfg=nemo_cfg,
            ),
        ),
        method=dict(
            gen_kwargs=dict(
                # only one beta supported by nemo generate for now
                beta=1,
            )
        ),
    )

    config = TRLConfig.update(default_config.to_dict(), hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])

    rewards = sum(dataset["train"]["reward"], [])
    eval_prompts = [prompt_output[0][0] for prompt_output in dataset["test"]["prompt_output"]][:280]
    reward_fn = create_reward_fn(device=torch.device("cpu"), use_steamshp=True)
    # reward_fn = create_reward_fn(device=torch.device("cpu"), use_steamshp=False)
    # reward_fn = create_reward_fn()
    reward_fn = lambda samples, prompts, outputs: [0.5 for _ in outputs]
    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

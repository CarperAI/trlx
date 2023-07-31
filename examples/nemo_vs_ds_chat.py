# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from math import floor
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
    default_ppo_config,
)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    cfg_name = os.environ.get("NEMO_CONFIG", "1.3B")
    if cfg_name == "1.3B":
        nemo_config = default_nemo_1_3b_config()
        batch_size = 16
        chunk_size = 128
        mini_batch_size = 16
        unfrozen_layers = -1

    elif cfg_name == "6.7B":
        nemo_config = default_nemo_1_3b_config()
        nemo_config.name = "megatron_gpt_6.7b"
        nemo_config.model.num_layers = 32
        nemo_config.model.hidden_size = 4096
        nemo_config.model.ffn_hidden_size = 16384
        nemo_config.model.num_attention_heads = 32
        batch_size = 4
        mini_batch_size = 4
        chunk_size = 16
        unfrozen_layers = -1

    elif cfg_name == "13B":
        nemo_config = default_nemo_1_3b_config()
        nemo_config.name = "megatron_gpt_13b"
        nemo_config.model.num_layers = 40
        nemo_config.model.hidden_size = 5120
        nemo_config.model.ffn_hidden_size = 20480
        nemo_config.model.num_attention_heads = 40
        nemo_config.model.tensor_model_parallel_size = 2
        batch_size = 16
        mini_batch_size = 4
        chunk_size = 16
        unfrozen_layers = -1

    elif cfg_name == "20B":
        nemo_config = default_nemo_1_3b_config()
        nemo_config.name = "megatron_gpt_20b"
        nemo_config.model.num_layers = 44
        nemo_config.model.hidden_size = 6144
        nemo_config.model.ffn_hidden_size = 24576
        nemo_config.model.num_attention_heads = 64

        nemo_config.model.tensor_model_parallel_size = 4
        batch_size = 16
        mini_batch_size = 2
        chunk_size = 16
        unfrozen_layers = -1

    elif cfg_name == "33B":
        nemo_config = default_nemo_1_3b_config()
        nemo_config.name = "megatron_gpt_33b"
        nemo_config.model.num_layers = 48
        nemo_config.model.hidden_size = 7168
        nemo_config.model.ffn_hidden_size = 28672
        nemo_config.model.num_attention_heads = 56

        nemo_config.trainer.num_nodes = 4
        nemo_config.trainer.devices = 8
        nemo_config.model.tensor_model_parallel_size = 8
        batch_size = 32
        mini_batch_size = 4
        chunk_size = 32
        unfrozen_layers = -1

    elif cfg_name == "66B":
        nemo_config = default_nemo_1_3b_config()
        nemo_config.trainer.num_nodes = 4
        nemo_config.trainer.devices = 8
        nemo_config.name = "megatron_gpt_66b"
        nemo_config.model.num_layers = 64
        nemo_config.model.hidden_size = 9216
        nemo_config.model.ffn_hidden_size = 36864
        nemo_config.model.num_attention_heads = 72

        nemo_config.model.tensor_model_parallel_size = 8
        batch_size = 32
        mini_batch_size = 2
        chunk_size = 32
        unfrozen_layers = 32

    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    config = default_config.evolve(
        train=dict(
            # set automatically
            total_steps=None,
            seq_length=512,
            batch_size=batch_size,
            minibatch_size=mini_batch_size,
            epochs=int(1e6),
            eval_interval=1e6,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=None,  # f"/mnt/hdd/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=1e6,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_ds_chat_benchmark",
            seed=2023,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "benchmark", cfg_name],
        ),
        optimizer=dict(
            name="distributed_fused_adam",
            kwargs=dict(
                lr=6.001e-5,
                weight_decay=1e-06,
                eps=1.0e-8,
                betas=(0.9, 0.95),
            ),
        ),
        scheduler=dict(
            name="CosineAnnealing",
        ),
        model=dict(num_layers_unfrozen=unfrozen_layers),
        method=dict(
            num_rollouts=chunk_size,
            init_kl_coef=0.05,
            scale_reward="ref",
            vf_coef=1,
            gen_kwargs=dict(temperature=1.0, max_new_tokens=256, min_new_tokens=256),
            chunk_size=chunk_size,
            ppo_epochs=1,
        ),
    )
    config.scheduler.kwargs = dict(warmup_steps=0, constant_steps=1e12, min_lr=6.0e-5)

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % 8

    reward_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    reward_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    reward_model.eval()
    reward_model.to("cpu")

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        reward_model.to(local_rank)
        mbs = max(1, config.method.chunk_size // 2)
        for i in range(0, len(samples) // mbs):
            inputs = reward_tokenizer(samples[i * mbs : (i + 1) * mbs], return_tensors="pt", padding=True)
            inputs = inputs.to(local_rank)
            with torch.no_grad():
                outputs = reward_model(**inputs)
            outputs.logits.cpu()
        reward_model.to("cpu")
        return [0.5 for _ in samples]

    # Take few words off of movies reviews as prompts
    dataset = load_dataset("Dahoas/rm-static", "train")
    dataset = dataset.shuffle(seed=2023)

    # select first 40% of the dataset
    dataset = dataset["train"].select(range(floor(len(dataset["train"]) * 0.4)))

    world_size = nemo_config.trainer.num_nodes * nemo_config.trainer.devices
    dp_world_size = world_size // (
        nemo_config.model.tensor_model_parallel_size * nemo_config.model.pipeline_model_parallel_size
    )
    global_batch_size = config.train.batch_size * dp_world_size
    config.train.total_steps = len(dataset) // global_batch_size

    print(f"Total steps: {config.train.total_steps=} {len(dataset)=} {global_batch_size=}")
    trlx.train(
        reward_fn=reward_fn,
        prompts=dataset["prompt"],
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

import json
import sys

from datasets import load_dataset

import trlx
from trlx.data.default_configs import (
    DPOConfig,
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
        epochs=100,
        total_steps=1000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateDPOTrainer",
        checkpoint_dir="checkpoints/dpo_hh",
    ),
    model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=1.0e-4)),  # train.total_steps
    method=DPOConfig(
        name="DPOConfig",
        gen_kwargs=dict(max_new_tokens=40, top_k=20, top_p=1.0, do_sample=True),
        beta=0.1,
        label_pad_token_id=-100,
        padding_value=0,
    ),
)


def preprocess(sample):
    sample["dpo"] = [sample["prompt"], sample["chosen"], sample["rejected"]]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)

    trlx.train(
        config=config,
        samples=dataset["train"]["dpo"],
        eval_prompts=dataset["test"]["prompt"][:280],
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

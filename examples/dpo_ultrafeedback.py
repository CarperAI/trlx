import itertools
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
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateDPOTrainer",
        checkpoint_dir="checkpoints/dpo_ultrafeedback",
    ),
    model=ModelConfig(model_path="HuggingFaceH4/mistral-7b-sft-beta", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="HuggingFaceH4/mistral-7b-sft-beta", truncation_side="right"),
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
    """
    Return list of lists with Context/Prompt at index 0, Chosen at index 1 and rejected at index 2
    """
    assert len(sample["chosen"]) == len(sample["rejected"]) == 2

    sample["dpo"] = [sample["prompt"], sample["chosen"][1]["content"], sample["rejected"][1]["content"]]
    return sample


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized").map(preprocess)

    trlx.train(
        config=config,
        samples=dataset["train_prefs"]["dpo"],
        eval_prompts=dataset["test_prefs"]["prompt"][:8],
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["User:", "user:", "Assistant:", "assistant:"]
        + ["{e}x {i}put:".format(e=e, i=i) for e, i in itertools.product(["e", "E"], ["in", "In", "out", "Out"])],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

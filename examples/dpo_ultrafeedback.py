import sys
import json
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer

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

model_path = "HuggingFaceH4/mistral-7b-sft-beta"
wandb_project = "trlx"

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=1,
        total_steps=70000,
        batch_size=1,
        checkpoint_interval=100000,
        eval_interval=5000,
        seed=42,
        project_name=wandb_project,
        pipeline="PromptPipeline",
        trainer="AccelerateDPOTrainer",
        checkpoint_dir="checkpoints/dpo_ultrafeedback",
    ),
    model=ModelConfig(model_path=model_path, num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path=model_path, truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=5e-7, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=1.0e-5)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=1.0e-4)),  # train.total_steps
    method=DPOConfig(
        name="DPOConfig",
        gen_kwargs=dict(max_new_tokens=768, temperature=0.7, top_k=50, top_p=0.95, do_sample=True),
        beta=0.1,
        label_pad_token_id=-100,
        padding_value=0,
    ),
)


def preprocess(sample, tokenizer, test=False):
    """
    Formats the input to the same training style used for mistral-7b-v0.1
    When fine-tuning, modify your pre-processing to match the prompt template used during pretraining.
    """
    assert len(sample["chosen"]) == len(sample["rejected"]) == 2

    assistant_prompt = "<|assistant|>"

    prompt, chosen = tokenizer.apply_chat_template(sample["chosen"], tokenize=False).split(assistant_prompt)
    rejected = tokenizer.apply_chat_template(sample["rejected"], tokenize=False).split(assistant_prompt)[-1]

    return {
        "prompt": prompt if not test else prompt + assistant_prompt,
        "chosen": assistant_prompt + chosen,
        "rejected": assistant_prompt + rejected,
    }


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    dataset["dpo_train"] = dataset["train_prefs"].map(
        partial(preprocess, tokenizer=tokenizer, test=False),
        remove_columns=["prompt_id", "score_chosen", "score_rejected", "messages"],
    )
    dataset["dpo_test"] = dataset["test_prefs"].map(
        partial(preprocess, tokenizer=tokenizer, test=True),
        remove_columns=["prompt_id", "score_chosen", "score_rejected", "messages"],
    )

    print(
        f"Length of training dataset : {len(dataset['dpo_train'])} \
            Length of test dataset : {len(dataset['dpo_test'])}"
    )

    trlx.train(
        config=config,
        samples=dataset["dpo_train"],
        eval_prompts=dataset["dpo_test"]["prompt"][:2],  # running eval on subset only
        stop_sequences=["<|user|>", "<|User|>"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

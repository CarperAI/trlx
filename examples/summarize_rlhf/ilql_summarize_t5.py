import os

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from transformers import AutoTokenizer

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
        seq_length=550,
        batch_size=8,
        epochs=100,
        total_steps=5000,
        checkpoint_interval=10000,
        eval_interval=1000,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="ilql_summarize_t5",
    ),
    model=ModelConfig(model_path="pvduy/flant5-xl_openai_tldr_sft", num_layers_unfrozen=-1, model_arch_type="seq2seq"),
    tokenizer=TokenizerConfig(tokenizer_path="pvduy/flant5-xl_openai_tldr_sft", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=5000, eta_min=1e-6)),
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
        gen_kwargs=dict(max_new_tokens=50, top_k=50, beta=[1, 2, 3], temperature=1.0),
    ),
)

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "CarperAI/openai_summarize_tldr_sft"


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel(SFT_MODEL_PATH)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)

    def reward_fn(samples):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def preprocess(sample):
        sample["prompt_output"] = [
            [sample["prompt"] + " TL;DR:", sample["chosen"][7:]],
            [sample["prompt"] + " TL;DR:", sample["rejected"][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    dataset["train"] = dataset["train"]
    dataset = dataset.map(preprocess)

    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])
    val_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    eval_prompts = list(val_dataset["prompt"])[:1000]

    trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=lambda samples, **kwargs: {"rewards": reward_fn(samples)},
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

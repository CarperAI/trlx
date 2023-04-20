import json
import math
import os
import sys
from itertools import islice
from typing import List

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_hh",
    ),
    model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
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


config_name = os.environ.get("CONFIG_NAME", "CC-7B")
if config_name == "125M":
    default_config.train.batch_size = 32
    default_config.train.total_steps = 1500
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_125M"
    default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.num_rollouts = 128
elif config_name == "1B":
    default_config.train.batch_size = 8
    default_config.train.total_steps = 2500
    default_config.optimizer.kwargs["lr"] = 6e-6
    default_config.scheduler.kwargs["eta_min"] = 6e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_1B"
    default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.chunk_size = 16
elif config_name == "6B":
    default_config.train.batch_size = 4
    default_config.train.seq_length = 512
    default_config.train.total_steps = 6000
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_6B"
    default_config.model.model_path = "Dahoas/pythia-6B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.chunk_size = 16
elif config_name == "20B":
    default_config.train.seq_length = 512
    default_config.train.batch_size = 1
    default_config.train.total_steps = 8000
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_20B"
    default_config.model.model_path = "EleutherAI/gpt-neox-20b"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.num_rollouts = 16
    default_config.method.chunk_size = 4
    default_config.method.ppo_epochs = 2
elif config_name == "CC-7B":
    default_config.train.batch_size = 1
    # default_config.train.minibatch_size = 1
    default_config.train.seq_length = 2048
    default_config.total_steps = 6000
    default_config.train.checkpoint_dir = "checkpoints/mbs=1_ppo_cc_StableLM-7B"
    default_config.train.checkpoint_interval = 1000
    # default_config.model.model_path =  "StabilityAI/stablelm-tuned-alpha-7b"
    default_config.model.model_path = "/mnt/nvme/home/dakota/ckpts/stablelm/7B-sft-combined/checkpoint-8000/"
    default_config.tokenizer.tokenizer_path = "/mnt/nvme/home/dakota/stablelm_tokenizer"
    # default_config.tokenizer.tokenizer_path = default_config.model.model_path
    default_config.method.chunk_size = 4
elif config_name == "CC-3B":
    default_config.train.batch_size = 1
    # default_config.train.minibatch_size = 1
    default_config.train.seq_length = 2048
    default_config.total_steps = 6000
    default_config.train.checkpoint_dir = "checkpoints/mbs=1_ppo_cc_StableLM-3B"
    default_config.train.checkpoint_interval = 1000
    default_config.model.model_path = "StabilityAI/stablelm-tuned-alpha-3b"
    default_config.tokenizer.tokenizer_path = default_config.model.model_path
    default_config.method.chunk_size = 4


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_reward_fn():  # noqa:  C901
    if os.environ.get("RANK", "0") == "0":

        class GPTRewardModel(nn.Module):
            def __init__(self):
                super().__init__()

                model = AutoModelForCausalLM.from_pretrained("CarperAI/vicuna-13b-fine-tuned", use_auth_token=True)
                self.config = model.config
                self.config.n_embd = (
                    self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
                )
                self.transformer = model.model
                self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
                self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned", use_auth_token=True)
                self.PAD_ID = self.tokenizer.pad_token_id

            def forward(
                self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                mc_token_ids=None,
                labels=None,
                return_dict=False,
                output_attentions=False,
                output_hidden_states=False,
            ):
                loss = None
                transformer_outputs = self.transformer(
                    input_ids,
                    attention_mask=attention_mask,
                )

                hidden_states = transformer_outputs[0]

                chosen_rewards = self.v_head(hidden_states).squeeze(-1)
                chosen_end_scores = []

                chosen = input_ids
                bs = input_ids.shape[0]
                for i in range(bs):
                    c_inds = (chosen[i] == self.PAD_ID).nonzero()
                    c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                    chosen_end_scores.append(chosen_rewards[i, c_ind - 1])

                inference = True

                rejected_end_scores = []

                if not inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    rejected_end_scores = torch.stack(rejected_end_scores)

                if inference:
                    chosen_end_scores = torch.stack(chosen_end_scores)
                    return {"chosen_end_scores": chosen_end_scores}

                return {
                    "loss": loss,
                    "chosen_end_scores": chosen_end_scores,
                    "rejected_end_scores": rejected_end_scores,
                }

        rw_tokenizer = AutoTokenizer.from_pretrained("CarperAI/vicuna-13b-fine-tuned", use_auth_token=True)
        rw_tokenizer.padding_side = "right"
        rw_device = torch.cuda.device_count() - 1
        rw_model = GPTRewardModel()
        rw_model.load_state_dict(torch.load("/mnt/hdd/duyphung/oa_rm_llama.pt")["module"])
        rw_model.transformer = rw_model.transformer
        rw_model.half().to(rw_device)
        rw_model.requires_grad_(False)
        rw_model.eval()

        @torch.inference_mode()
        def get_scores(samples: List[str], batch_size=4):
            scores_list = []
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = [chosen for chosen in sub_samples]
                encodings_dict = rw_tokenizer(
                    sub_samples,
                    truncation=True,
                    max_length=2048,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to(rw_device)
                attn_masks = encodings_dict["attention_mask"].to(rw_device)
                with torch.no_grad():
                    sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
                scores_list.append(sub_scores["chosen_end_scores"])
            scores = torch.cat(scores_list, dim=0)
            return scores

        def reward_fn(samples, **kwargs):
            samples = [
                s.replace("<|SYSTEM|>", "")
                .replace("<|USER|>", "\n### Human: ")
                .replace("<|ASSISTANT|>", "\n### Assistant: ")
                for s in samples
            ]
            return get_scores(samples)

    else:
        reward_fn = True

    return reward_fn


def chatcombined_prompt_response():
    dataset = load_dataset("dmayhem93/ChatCombined", split="train")

    def split_prompt_response(example):
        split = example["text"].split("<|ASSISTANT|>")
        if len(split) < 2:
            return None
        prompt, response, *rest = split
        return {"prompt": prompt + "<|ASSISTANT|>", "chosen": response}

    return dataset.filter(lambda x: split_prompt_response(x) is not None).map(split_prompt_response)


def main(hparams={}):
    tags = ["not-delta-reward"]
    config = TRLConfig.update(default_config, hparams)
    config.train.tags = tags

    reward_fn = create_reward_fn()

    dataset = chatcombined_prompt_response()
    prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset, 280)]
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset, 280)]

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["<|USER|>"],
    ).save_pretrained(f"mbs1-{config_name}-chatcombined-sft-vicuna-rlhf")


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

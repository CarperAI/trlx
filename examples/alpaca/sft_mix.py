import json
import os
from argparse import ArgumentParser
from typing import Dict, List
from functools import partial

import torch

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config

# Patch GPTNeoxForCausalLM's GPTNeoxAttention to use Flash Attention
# by using torch.nn.functional.scaled_dot_product_attention

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXForCausalLM


def _attn_flash(self, query, key, value, attention_mask=None, head_mask=None):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    query = query.type_as(value)
    key = key.type_as(value)
    attention_mask = None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, is_causal=attention_mask is None
    )
    return attn_output, None

def patch_gpt_neox_attention(m):
    if isinstance(m, GPTNeoXAttention):
        m._attn = _attn_flash.__get__(m, m.__class__)

def unpatch_gpt_neox_attention(m):
    if isinstance(m, GPTNeoXAttention):
        m._attn = GPTNeoXAttention._attn.__get__(m, m.__class__)

def preprocess(instruction: str, input: str, output: str):
    """Build Alpaca prompt and output from instruction and input/output examples"""
    if input:
        prefix = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request."
        )
        prompt = f"{prefix}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        return [prompt, output]
    else:
        prefix = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )
        prompt = f"{prefix}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        return [prompt, output]


def main(hparams={}, model_name="EleutherAI/gpt-j-6B", dataset="tatsu-lab/alpaca", tokenizer_path=None):
    tags = [model_name, dataset]

    model = GPTNeoXForCausalLM.from_pretrained(model_name, use_auth_token=True)

    model.apply(patch_gpt_neox_attention)
    # model.gpt_neox = torch.compile(model.gpt_neox) 

    if tokenizer_path is None:
        tokenizer_path = model_name

    config = default_sft_config()
    config = config.evolve(
        train=dict(
            total_steps=1200,
            batch_size=16,
            seq_length=4096,
            minibatch_size=4,
            tags=tags,
        ),
        model=dict(
            model_path=model,
        ),
        tokenizer=dict(
            tokenizer_path=tokenizer_path,
        ),
        optimizer=dict(kwargs=dict(lr=2e-5, weight_decay=0.0)),
        scheduler=dict(kwargs=dict(eta_min=2e-5)),
        method=dict(
            gen_kwargs=dict(
                max_new_tokens=256,
            )
        ),
    )

    # Merge sweep config with default config if given
    config = TRLConfig.update(config.to_dict(), hparams)

    # alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = load_dataset(dataset, split="train")
    alpaca = [preprocess(x["instruction"], x["input"], x["output"]) for x in alpaca]

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):

        trainer = trlx.train(
            samples=alpaca,
            eval_prompts=["User:", "User: "],  # zs_rewrite,
            metric_fn=None,  # metric_fn,
            config=config,
        )

    slug = f"{model_name.split('/')[-1]}-{dataset.split('/')[-1]}"
    trainer.save_pretrained(f"{slug}-mix-sft")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("override_hparams", type=str, default="{}", nargs="?")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-6.9b")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--tokenizer_path", type=str, default=None)

    args = parser.parse_args()
    hparams = json.loads(args.override_hparams)

    main(hparams, args.model_name, args.dataset, args.tokenizer_path)

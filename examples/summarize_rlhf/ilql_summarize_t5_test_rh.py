import os
import pandas as pd
import json
import smart_open
import datetime
from datetime import date

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from transformers import AutoTokenizer

import trlx
import evaluate
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

# timestamp = date.strftime(datetime.datetime.now(), "%Y-%m-%d")
# print(timestamp)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        batch_size=1,
        epochs=1,
        total_steps=10000,
        checkpoint_interval=1250,
        eval_interval=1250,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="/root/.neeva/data/ilql_summarize_11b_t5_ckpt",
    ),
    model=ModelConfig(model_path="/data/models/11b-checkpoint-75614", num_layers_unfrozen=-1, model_arch_type="seq2seq"),
    tokenizer=TokenizerConfig(tokenizer_path="/data/models/11b-checkpoint-75614", truncation_side="right"),
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

def _truncate_text_to_num_tokens(text, max_length, tokenizer):
    length = sum([len(t) for t in tokenizer.tokenize(text)[:max_length]])
    return text[:length]

def main(hparams={}):
    
    config = TRLConfig.update(default_config, hparams)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"

    
    # TODO change to the larger calibrated set
    dataset = pd.read_csv('/data/data/outputs-75614-10k.csv')
    prompts_outputs = []
    rewards = []
    for i, r in dataset.iterrows():
        if (type(r.response) == str) & (type(r.generation) == str):
            num_extra_tokens = len(tokenizer.tokenize(" Summary:")) + 2
            num_tokens_response = len(tokenizer.tokenize(r.response))
            num_tokens_generation = len(tokenizer.tokenize(r.generation))    
            prompts_outputs.append([_truncate_text_to_num_tokens(r.text, max_length, tokenizer) + " Summary:", r.response])
            rewards.append(1)
            prompts_outputs.append([_truncate_text_to_num_tokens(r.text, max_length, tokenizer) + " Summary:", r.generation])
            rewards.append(-1)
                          
    # add eval prompt
    valid = [json.loads(l) for l in smart_open.open('/data/data/valid.json').readlines()]
    eval_prompts = [i['text'] for i in valid][:100]
        
    trainer = trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=None,
        eval_prompts=eval_prompts,
        config=config,
    )
    trainer.save_pretrained("/data/ilql_summarize_t5_11b_output")


if __name__ == "__main__":
    import json
    import sys

    hparams = {} # if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

import sys
from typing import List

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from reward_model import GPTRewardModel
from summarize_dataset import get_dataset_from_jsonl
import trlx
from trlx.data.configs import TRLConfig
import argparse
import os
import wandb
import evaluate

rouge = evaluate.load('rouge')

if __name__ == "__main__":
    
    def reward_fn(samples: List[str]):
        articles = [ 
            sample.split("<sep>")[0].strip() for sample in samples
        ]
        summs = [
            sample.split("<sep>")[1].strip() for sample in samples
        ]
        labels = [
            prompt_label[sample] for sample in articles
        ]
    
        scores = [   
            rouge.compute(predictions=[summary], references=[label])
            for (summary, label) in zip(summs, labels)
        ]
        scores = [
           (score['rouge1'] + score['rouge2'] + score['rougeL']) / 3. 
                for score in scores
        ]
        return scores


    config = TRLConfig.load_yaml("ppo_config_cnn_daily.yml")
    
    from datasets import load_dataset
    dataset = load_dataset("cnn_dailymail", '3.0.0', split="validation")
    prompts = dataset["article"][0:100]#[dataset["article"][0]] * 1000
    summaries = dataset["highlights"][0:100]#[dataset["highlights"][0]] * 1000
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    

    prompt_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    for i in range(len(prompts)):
        # if prompts[i].startswith("London (CNN)At the time it probably seemed like fun: Jeremy Clarkson and"):
        #     import ipdb; ipdb.set_trace()
        key = tokenizer.decode(
            tokenizer(
                prompts[i],
                truncation=True,
                max_length=max_length
            )['input_ids'],
            skip_special_tokens=True, 
        )
        prompt_label[key.strip()] = summaries[i]
    
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=prompts[0:100],
        config=config
    )
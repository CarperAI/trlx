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


#wandb.init(project="trlx_ver2", name="trlx-gpt2-summarize-val-test", entity="pvduy")

if __name__ == "__main__":
    
    # rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # rw_tokenizer.pad_token = rw_tokenizer.eos_token
    # rw_model = GPTRewardModel("/fsx/home-duyphung/sandbox/refactor_summarize_rlhf/trlx/examples/summarize_rlhf/gptneo-supervised-summarize-checkpoint/checkpoint-1000")
    # rw_model.load_state_dict(torch.load("reward_model_inspect/ckpts/openai_comparison_summary/gpt-j/checkpoint-1700/pytorch_model.bin"))
    # rw_model.half()
    # rw_model.eval()
    # rw_device = torch.device("cuda:{}".format(1))
    # rw_model.to(rw_device)
    
    def reward_fn(samples: List[str]):
        
        return torch.tensor([.5] * len(samples))
        # original_samples = [text.split('TL;DR:')[0] + 'TL;DR: ' for text in samples]
        # original_samples = [text + train_post_summ[text] for text in original_samples]
        
        # ori_lst_scores = []
        # batch_size = 2
        # for i in range(0, len(original_samples), batch_size):
        #     sub_samples = original_samples[i:i+batch_size]
        #     sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
        #     encodings_dict = rw_tokenizer(
        #             sub_samples, 
        #             truncation=True, 
        #             max_length=550, 
        #             padding="max_length",
        #             return_tensors="pt"
        #     )
        #     input_ids = encodings_dict['input_ids'].to(rw_device)
        #     attn_masks = encodings_dict['attention_mask'].to(rw_device)
        #     input_ids = input_ids.repeat(2, 1)
        #     attn_masks = attn_masks.repeat(2, 1)
        #     with torch.no_grad():
        #         sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        #     ori_lst_scores.append(sub_scores['chosen_end_scores'])
        # ori_scores = torch.cat(ori_lst_scores, dim=0)
        
        # lst_scores = []
        # batch_size = 2
        # for i in range(0, len(samples), batch_size):
        #     sub_samples = samples[i:i+batch_size]
        #     sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
        #     encodings_dict = rw_tokenizer(
        #             sub_samples, 
        #             truncation=True, 
        #             max_length=550, 
        #             padding="max_length",
        #             return_tensors="pt"
        #     )
        #     input_ids = encodings_dict['input_ids'].to(rw_device)
        #     attn_masks = encodings_dict['attention_mask'].to(rw_device)
        #     input_ids = input_ids.repeat(2, 1)
        #     attn_masks = attn_masks.repeat(2, 1)
        #     with torch.no_grad():
        #         sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        #     lst_scores.append(sub_scores['chosen_end_scores'])
        # scores = torch.cat(lst_scores, dim=0)
        # norms_scores = scores  - ori_scores
        # return norms_scores

    train_openai_summ, train_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "train.jsonl"), False)
    val_openai_summ, val_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "valid.jsonl"), False)
    test_openai_sum, test_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "test.jsonl"), False)
    
    # train_post_summ = {}
    # for i in range(len(train_openai_summ)):
    #     tmp = rw_tokenizer.decode(rw_tokenizer(train_openai_summ[i])['input_ids'])
    #     train_post_summ[tmp] = train_labels[i]
    
    # for i in range(len(val_openai_summ)):
    #     tmp = rw_tokenizer.decode(rw_tokenizer(val_openai_summ[i])['input_ids'])
    #     train_post_summ[tmp] = val_labels[i]

    # for i in range(len(test_openai_sum)):
    #     tmp = rw_tokenizer.decode(rw_tokenizer(test_openai_sum[i])['input_ids'])
    #     train_post_summ[tmp] = test_labels[i]

    # Take few words off of movies reviews as prompts
    from datasets import load_dataset

    imdb = load_dataset("imdb", split="train+test")
    prompts = ["Generate positive review for IMDB film start with: " + " ".join(review.split()[:4]) for review in imdb["text"]]


    #prompts = val_openai_summ
    print(len(prompts))
    config = TRLConfig.load_yaml("ppo_config_summ_t5.yml")
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=prompts[0:10],
        config=config
    )

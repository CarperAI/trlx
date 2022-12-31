import os

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from reward_model import GPTRewardModel
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import argparse

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    
def create_comparision_dataset(path):
    
    def make_text(post, summarize):
        return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"
    
    with open(path, 'r') as f:
        dataset = [json.loads(line) for line in f]
        
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        post = sample['info']
        chosen_summary = sample['summaries'][sample['choice']]['text']
        rejected_summary = sample['summaries'][1 - sample['choice']]['text']
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair['chosen'] = make_text(post, chosen_summary)
        pair['rejected'] = make_text(post, rejected_summary)
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer('<|startoftext|>' + chosen + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            rejected_encodings_dict = tokenizer('<|startoftext|>' + rejected + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]


class DataCollatorReward:
    
    def __call__(self, data):
        batch = {}
        batch['input_ids'] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch['attention_mask'] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch['labels'] = torch.tensor([0]*len(data) + [1] * len(data))
        return batch

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

    model = GPTRewardModel("../gptneo-supervised-summarize-checkpoint/checkpoint-1000")
    # layers = model.transformer.h
    # num_layers = len(layers)
    # num_unfrozen = int(0.75 * num_layers)
    # for layer in layers[:-num_unfrozen]:
    #     layer.requires_grad_(False)
    model.load_state_dict(torch.load("ckpts/openai_comparison_summary/gpt-j/checkpoint-1700/pytorch_model.bin"))
    max_length = 550
    val_pairs = create_comparision_dataset(os.path.join("../../../../openai_data/comparisons", "valid_comparisons.jsonl"))
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    from torch.utils.data import DataLoader
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward()
    )
    model.cuda()
    model.eval()
    model.half()
    correct = 0
    lst_chosen = []
    lst_reject = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = model(**batch)
            correct += sum(outputs['chosen_end_scores'] > outputs['rejected_end_scores'])
            lst_chosen.append(outputs['chosen_end_scores'].cpu())
            lst_reject.append(outputs['rejected_end_scores'].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))

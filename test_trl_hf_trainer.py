import warnings
import torch
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import random_split

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline, TrainingArguments, IntervalStrategy

from trl.gpt2 import GPT2HeadWithValueModel
from ppo_hf_trainer import PPOTrainer
import time
#os.environ["WANDB_DISABLED"] = "true"  # TODO(dahoas): remove

# Training config
config = {
	"model_name": "gpt2",
	"cls_model_name": "gpt2",
	"steps": 20000,
	"batch_size": 16,
	"forward_batch_size": 16,
	"txt_in_min_len": 2,
	"txt_in_max_len": 8,
	"txt_out_min_len": 4,
	"txt_out_max_len": 16,
}

# Setup reward function
pipe_device = 0 if torch.cuda.is_available() else -1
sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)
sent_kwargs = {
	"return_all_scores": True,
	"function_to_apply": "none",
	"batch_size": config["forward_batch_size"]
}

# Setup length sampler
class LengthSampler:
	def __init__(self, min_value, max_value):
		self.values = list(range(min_value, max_value))
	def __call__(self):
		return 5
		#return np.random.choice(self.values)

input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

# Model setup
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name']).cuda()  # Need to place model on gpu b/c not handled by HF Trainer
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model.model_parallel = True  # Set true for trainer

# Tokenizer setup
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"  # Note: This is important otherwise generating continuation from endoftext token
def tokenize(sample):
	encoding = gpt2_tokenizer(sample["text"], padding='max_length', max_length=input_size(), truncation=True)
	sample['input_ids'] = encoding['input_ids']
	sample['attention_mask'] = encoding['attention_mask']
	#print(sample)
	return sample

# load imdb with datasets
ds = load_dataset('imdb', split='train')
ds = ds.filter(lambda x: len(x["text"])>200, batched=False)
ds = ds.map(tokenize, batched=False)

def collator(data):
	return {key: torch.stack([torch.tensor(d[key]) for d in data]) for key in ['input_ids', 'attention_mask']}

train_size = int(0.95 * len(ds))
train_dataset, eval_dataset = random_split(ds, [train_size, len(ds) - train_size])

#gpt_blocks = list(gpt2_model.transformer.h)[:-1]
#for m in gpt_blocks:
#	for p in m.parameters():
#		p.requires_grad = False

# Set configs
ppo_config = {
	'adap_kl_ctrl': True,
	"init_kl_coef":0.2,
	"target": 6,
	"horizon":10000,
	"gamma":1,
	"lam":0.95,
	"cliprange": .2,
	"cliprange_value":.2,
	"vf_coef":.1,
	"ppo_epochs": 4,  # Currently not implemented
}

gen_kwargs = {
	"max_length": 64,
    "min_length": 20,
	"pad_token_id": gpt2_tokenizer.eos_token_id
}

training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=10, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=16, per_device_eval_batch_size=16, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True)

# Set up trainer
trainer = PPOTrainer(model=gpt2_model, args=training_args, train_dataset=train_dataset,
		eval_dataset=eval_dataset, data_collator=collator)
trainer.init_ppo_params(ppo_config, sent_kwargs, gen_kwargs, gpt2_model_ref, gpt2_tokenizer, sentiment_pipe)
trainer.train()



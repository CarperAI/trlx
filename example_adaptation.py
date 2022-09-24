# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

import warnings
import torch
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from accelerate import Accelerator
from torch.optim import Adam
import random
import pdb
from transformers import AutoModelForCausalLM

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo_deepspeed import PPOTrainer
import time
from transformers import DataCollatorForLanguageModeling

from trl.core import (logprobs_from_logits,
					  whiten,
					  clip_by_value,
					  entropy_from_logits,
					  flatten_dict,
					  average_torch_dicts,
					  stats_to_np,
					  stack_dicts,
					  add_suffix,
					  WANDB_PADDING)

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def get_dataloaders(accelerator: Accelerator, ppo_config):
	"""
	Creates a set of `DataLoader`s for the `glue` dataset,
	using "bert-base-cased" as the tokenizer.
	Args:
		accelerator (`Accelerator`):
			An `Accelerator` object
		batch_size (`int`, *optional*):
			The batch size for the train and validation DataLoaders.
	"""
	ds = load_dataset('imdb', split='train')
	ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
	ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

	gpt2_tokenizer = AutoTokenizer.from_pretrained(ppo_config['model_name'])
	gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
	ppo_config['gen_kwargs']['pad_token_id'] = gpt2_tokenizer.eos_token_id

	gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
	gpt2_tokenizer.padding_side = "left"

	def tokenize(sample):
		encoding = gpt2_tokenizer(sample["review"], padding='max_length', max_length=ppo_config['input_size'], truncation=True)
		sample["tokens"] = encoding['input_ids']
		sample["attention_mask"] = encoding['attention_mask']
		sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
		return sample

	ds = ds.map(tokenize, batched=False)

	def collater(data):
		return dict((key, [d[key] for d in data]) for key in data[0])

	train_dataloader = torch.utils.data.DataLoader(ds, batch_size=ppo_config['batch_size'], collate_fn=collater)

	return train_dataloader, ppo_config, gpt2_tokenizer


def training_function(ppo_config):
	# Initialize accelerator
	accelerator = Accelerator()
	# Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
	lr = ppo_config["lr"]
	num_epochs = int(ppo_config["steps"]) // int(ppo_config['batch_size'])
	batch_size = int(ppo_config["batch_size"])

	# If the batch size is too big we use gradient accumulation
	#gradient_accumulation_steps = 1
	#if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
	#    gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
	#    batch_size = MAX_GPU_BATCH_SIZE

	train_dataloader, ppo_config, gpt2_tokenizer = get_dataloaders(accelerator, ppo_config)
	# Instantiate the model (we build the model here so that the seed also control new weights initialization)
	gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')

	# We could avoid this line since the accelerator is set with `device_placement=True` (default value).
	# Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
	# creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
	gpt2_model = gpt2_model.to(accelerator.device)

	# Instantiate optimizer
	optimizer = AdamW(params=gpt2_model.parameters(), lr=lr)

	# Instantiate scheduler
	lr_scheduler = get_linear_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=100,
		num_training_steps=(len(train_dataloader) * num_epochs),
	)

	# Prepare everything
	# There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
	# prepare method.
	gpt2_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
		gpt2_model, optimizer, train_dataloader, lr_scheduler
	)

	text_in = "temp"
	dummy_input = gpt2_tokenizer(text_in, return_tensors="pt").to(accelerator.device)

	unwrapped_model = accelerator.unwrap_model(gpt2_model)

	# Now we train the model
	device = accelerator.device
	gpt2_model.train()
	for step, batch in enumerate(train_dataloader):
		# We could avoid this line since we set the accelerator with `device_placement=True`.
		query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
		#### Get response from gpt2
		t = time.time()
		response_tensors = []
		for i in range(ppo_config['batch_size']):
			gen_len = ppo_config['gen_size']
			with torch.no_grad():
				# This seems to be required even for deepspeed
				_ = gpt2_model(**dummy_input)
				query_tensor = query_tensors[i]
				response = unwrapped_model.generate(query_tensor.unsqueeze(dim=0), synced_gpus=True, **ppo_config['gen_kwargs'])
				response = response[:, query_tensor.size()[0] : query_tensor.size()[0] + gen_len]
			response_tensors.append(response.squeeze())

		batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
		query_tensors = torch.cat([torch.tensor(t).long().to(accelerator.device) for t in batch["tokens"]])
		outputs = gpt2_model(query_tensors)
		loss = torch.sum(outputs.logits)
		accelerator.backward(loss)
		optimizer.step()
		lr_scheduler.step()
		optimizer.zero_grad()



if __name__ == "__main__":
	ppo_config = {
		"model_name": "lvwerra/gpt2-imdb",
		"cls_model_name": "lvwerra/distilbert-imdb",
		"steps": 20000,
		"batch_size": 16,
		"forward_batch_size": 16,
		"ppo_epochs": 4,
		"input_size": 5,
		"gen_size": 12,
		"lr": 5.0e-6,
		"init_kl_coef":0.2,
		"target": 6,
		"horizon":10000,
		"gamma":1,
		"lam":0.95,
		"cliprange": .2,
		"cliprange_value":.2,
		"vf_coef":.2,
		'sent_kwargs': {
			"return_all_scores": True,
			"function_to_apply": "none",
			"batch_size": 16
		},
		"gen_kwargs": {
			"max_length": 64,
			"min_length": 20,
			"top_k": 0.0,
			"top_p": 1.0,
			"do_sample": True,
		}
	}
	training_function(ppo_config)
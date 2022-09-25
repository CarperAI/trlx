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
	accelerator = Accelerator(log_with='wandb')
	accelerator.init_trackers('trl_accelerate', config=ppo_config)
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
	data_collator = DataCollatorForLanguageModeling(gpt2_tokenizer, mlm=False)
	# Instantiate the model (we build the model here so that the seed also control new weights initialization)
	gpt2_model = GPT2HeadWithValueModel.from_pretrained(ppo_config['model_name'])
	gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(ppo_config['model_name'])

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

	pipe_device = 0 if torch.cuda.is_available() else -1
	sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)

	# Now we train the model
	device = accelerator.device
	gpt2_model.train()
	for step, batch in tqdm(enumerate(train_dataloader)):
		# We could avoid this line since we set the accelerator with `device_placement=True`.
		query_tensors = torch.stack([torch.tensor(t).long().to(device) for t in batch["tokens"]])
		for query_tensor in query_tensors:
			assert query_tensor.size()[0] == ppo_config['input_size']
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
		response_tensors = torch.stack(response_tensors)
		
		#### Compute sentiment score
		texts = [q + r for q,r in zip(batch['query'], batch['response'])]
		# Ouptut is a list of lists of containing two dictionaries corresponding to pos/neg class and score
		# may be dependent on the hf version
		pipe_outputs = sentiment_pipe(texts, **ppo_config['sent_kwargs'])
		scores = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
		mean_score = torch.mean(scores).item()
		accelerator.log({"mean_score": mean_score}, step=step)

		# Run PPO step
		bs = ppo_config['batch_size']
		assert bs == len(query_tensors), f"Batch size ({bs}) does not match number of examples ({len(query_tensors)})"

		# batched forward pass
		fbs = ppo_config['forward_batch_size']
		all_logprobs, all_ref_logprobs, all_values = [], [], []
		for i in range(int(bs/fbs)):
			query_batch = query_tensors[i*fbs:(i+1)*fbs]
			response_batch = response_tensors[i*fbs:(i+1)*fbs]
			input_ids = data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])["input_ids"]
			with torch.no_grad():
				logits, _, v = gpt2_model(input_ids)
				ref_logits, _, _ = gpt2_model_ref(input_ids.cpu()) # TODO(dahoas): Need to make decision about what to do with ref model: keep on cpu?
				ref_logits = ref_logits.to(device)
			logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
			ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])
			for j in range(fbs):
				start = len(query_batch[j])-1
				end = len(query_batch[j]) + len(response_batch[j])-1
				all_values.append(v[j, start-1:end-1])
				all_logprobs.append(logprobs[j, start:end])
				all_ref_logprobs.append(ref_logprobs[j, start:end])

		# Compute rewards
		all_rewards, non_score_rewards = [], []
		for score, logprob, ref_logprob in zip(scores, all_logprobs, all_ref_logprobs):
			kl = logprob - ref_logprob
			non_score_reward = -ppo_config['init_kl_coef'] * kl
			non_score_rewards.append(non_score_reward)
			reward = non_score_reward.clone()
			reward[-1] += score
			all_rewards.append(reward)

		# Train minibatches
		idxs = list(range(bs))
		for _ in range(ppo_config['ppo_epochs']):
			random.shuffle(idxs)
			for i in range(bs):
				idx = idxs[i]
				old_logprob = all_logprobs[idx].unsqueeze(0)
				values = all_values[idx].unsqueeze(0)
				rewards = all_rewards[idx].unsqueeze(0)
				query = query_tensors[idx].unsqueeze(0)
				response = response_tensors[idx].unsqueeze(0)
				model_input = torch.cat([query.squeeze(), response.squeeze()]).unsqueeze(0)

				# Compute loss
				lastgaelam = 0
				advantages_reversed = []
				gen_len = response.shape[1]

				for t in reversed(range(gen_len)):
					nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
					delta = rewards[:, t] + ppo_config['gamma'] * nextvalues - values[:, t]
					lastgaelam = delta + ppo_config['gamma'] * ppo_config['lam'] * lastgaelam
					advantages_reversed.append(lastgaelam)
				advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

				returns = advantages + values
				advantages = whiten(advantages)
				advantages = advantages.detach()

				# Q: How is this logprob different from old_logprobs
				logits, _, vpred = gpt2_model(model_input)
				logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

				#only the generation part of the values/logprobs is needed
				logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

				vpredclipped = clip_by_value(vpred,
											values - ppo_config["cliprange_value"],
											values + ppo_config["cliprange_value"])

				vf_losses1 = (vpred - returns)**2
				vf_losses2 = (vpredclipped - returns)**2
				vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
				vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

				ratio = torch.exp(logprob - old_logprob)

				pg_losses = -advantages * ratio
				pg_losses2 = -advantages * torch.clamp(ratio,
													1.0 - ppo_config['cliprange'],
													1.0 + ppo_config['cliprange'])

				pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
				pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

				loss = pg_loss + ppo_config['vf_coef'] * vf_loss

				# Compute dummy loss and backprop
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
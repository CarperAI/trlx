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
import argparse

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo_deepspeed import PPOTrainer
import time
from transformers import DataCollatorForLanguageModeling
import yaml

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


# Ignore warnings
#warnings.filterwarnings("ignore") # TODO(dahoas): remove

def run(ppo_config):
	# Adding accelerator
	accelerator = Accelerator(log_with='wandb')
	accelerator.init_trackers('trl_accelerate', config=ppo_config)
	device = accelerator.device
	print("DEVICE: ", device)

	# load imdb with datasets
	ds = load_dataset('imdb', split='train')
	ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
	ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

	pipe_device = -1
	sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)

	gpt2_tokenizer = AutoTokenizer.from_pretrained(ppo_config['model_name'])
	gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

	gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
	gpt2_tokenizer.padding_side = "left"
	ppo_config['gen_kwargs']['pad_token_id'] = gpt2_tokenizer.eos_token_id
	def tokenize(sample):
		encoding = gpt2_tokenizer(sample["review"], padding='max_length', max_length=ppo_config['input_size'], truncation=True)
		sample["tokens"] = encoding['input_ids']
		sample["attention_mask"] = encoding['attention_mask']
		sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
		return sample

	ds = ds.map(tokenize, batched=False)

	def collater(data):
		return dict((key, [d[key] for d in data]) for key in data[0])

	dataloader = torch.utils.data.DataLoader(ds, batch_size=ppo_config['batch_size'], collate_fn=collater)

	gpt2_model = GPT2HeadWithValueModel.from_pretrained(ppo_config['model_name'])
	gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(ppo_config['model_name'])

	optimizer = Adam(gpt2_model.parameters(), lr=ppo_config['lr'])
	total_ppo_epochs = int(np.ceil(ppo_config["steps"]/ppo_config['batch_size']))
	#scheduler = torch.optim.lr_scheduler.CosieAnnealingLR(optimizer, total_ppo_epochs) this should not be needed

	data_collator = DataCollatorForLanguageModeling(gpt2_tokenizer, mlm=False)

	gpt_blocks = list(gpt2_model.transformer.h)[:-ppo_config['num_layers_unfrozen']]
	for m in gpt_blocks:
		for p in m.parameters():
			p.requires_grad = False



	# Prepare accelerator
	gpt2_model, optimizer, dataloader = accelerator.prepare(gpt2_model, optimizer, dataloader)

	text_in = "temp"
	dummy_input = gpt2_tokenizer(text_in, return_tensors="pt").to(accelerator.device)

	# had to run this 1 time at the start else was giving device mismatch error.
	# So, before directly using `model.generate` pass a batch with dummy data through the model
	# outputs = gpt2_model(**dummy_input)
	unwrapped_model = accelerator.unwrap_model(gpt2_model)

	print("NUM EPOCHS: ", total_ppo_epochs) if accelerator.is_main_process else None
	for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader)), disable=not accelerator.is_local_main_process):
		gpt2_model.train()
		query_tensors = torch.stack([torch.tensor(t).long().to(device) for t in batch["tokens"]])
		#### Get response from gpt2
		gen_len = ppo_config['gen_size']
		with torch.no_grad():
			# This seems to be required even for deepspeed
			dummy_outputs = gpt2_model(**dummy_input)
			response = unwrapped_model.generate(query_tensors, synced_gpus=True, **ppo_config['gen_kwargs'])
			response_tensors = response[:, query_tensors.size()[1] : query_tensors.size()[1] + gen_len]

		batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

		#### Compute sentiment score
		texts = [q + r for q,r in zip(batch['query'], batch['response'])]
		# Ouptut is a list of lists of containing two dictionaries corresponding to pos/neg class and score
		# may be dependent on the hf version
		pipe_outputs = sentiment_pipe(texts, **ppo_config['sent_kwargs'])
		scores = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
		mean_score = torch.mean(scores).item()
		accelerator.log({"mean_score": mean_score}, step=epoch)

		# Run PPO step
		bs = ppo_config['batch_size']
		assert bs == len(query_tensors), f"Batch size ({bs}) does not match number of examples ({len(query_tensors)})"
		response_lengths = [r.size()[0] for r in response_tensors]

		# batched forward pass
		all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
		assert all_tokens.size()[1] == query_tensors.size()[1] + response_tensors.size()[1]
		with torch.no_grad():
			logits, _, v = gpt2_model(all_tokens)
			ref_logits, _, _ = gpt2_model_ref(all_tokens.cpu()) # TODO(dahoas): Need to make decision about what to do with ref model: keep on cpu?
			ref_logits = ref_logits.to(device)
		logprobs = logprobs_from_logits(logits[:,:-1,:], all_tokens[:,1:])
		ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], all_tokens[:,1:])
		start = query_tensors.size()[1]-1
		end = query_tensors.size()[1] + response_tensors.size()[1] - 1
		all_values = v[:, start-1 : end-1]
		all_logprobs = logprobs[:, start : end]
		all_ref_logprobs = ref_logprobs[:, start : end]

		# Compute rewards
		kls = all_logprobs - all_ref_logprobs
		non_score_rewards = -ppo_config['init_kl_coef'] * kls
		all_rewards = non_score_rewards.clone()
		all_rewards[:, -1] += scores

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

				## Backprop loss
				accelerator.backward(loss)
				accelerator.wait_for_everyone()
				optimizer.step()
				optimizer.zero_grad()

		# TODO(dahoas): Update kl_ctl term
		#self.kl_ctl.update(stats['objective/kl'], ppo_config['batch_size'])

	accelerator.end_training()
	print("FINISHED TRAINING")

def load_yaml(path):
	with open(path, 'r') as f:
		config = yaml.safe_load(f)
	return config

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	ppo_config = load_yaml(args.config_path)
	ppo_config['sent_kwargs']['batch_size'] = ppo_config['batch_size']
	ppo_config['accelerate_config'] = load_yaml(ppo_config['accelerate_config_path'])
	run(ppo_config)

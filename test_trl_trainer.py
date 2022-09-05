import warnings
import torch
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from accelerate import Accelerator

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer
import time

'''
HF has fullly automatic support for deepspeed when using Trainer class.
Otherwise no reason to use HF with deepspeed really.
'''
# Ignore warnings
warnings.filterwarnings("ignore") # TODO(dahoas): remove

config = {
	"model_name": "gpt2",
	"cls_model_name": "gpt2",
	"steps": 20000,
	"batch_size": 16,
	"forward_batch_size": 4,
	"ppo_epochs": 4,
	"txt_in_min_len": 2,
	"txt_in_max_len": 8,
	"txt_out_min_len": 4,
	"txt_out_max_len": 16,
	"lr": 1.41e-5,
	"init_kl_coef":0.2,
	"target": 6,
	"horizon":10000,
	"gamma":1,
	"lam":0.95,
	"cliprange": .2,
	"cliprange_value":.2,
	"vf_coef":.1,
}

pipe_device = 0 if torch.cuda.is_available() else -1

# load imdb with datasets
ds = load_dataset('imdb', split='train')
ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

sent_kwargs = {
	"return_all_scores": True,
	"function_to_apply": "none",
	"batch_size": config["forward_batch_size"]
}

sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)

class LengthSampler:
	def __init__(self, min_value, max_value):
		self.values = list(range(min_value, max_value))
	def __call__(self):
		return 5
		#return np.random.choice(self.values)

input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])
#input_size = config["txt_in_max_len"]
#output_size = config["txt_out_max_len"]

gpt2_tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_size = "left"
def tokenize(sample):
	encoding = gpt2_tokenizer(sample["review"], padding='max_length', max_length=input_size(), truncation=True)
	sample["tokens"] = encoding['input_ids']
	sample["attention_mask"] = encoding['attention_mask']
	sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
	return sample

ds = ds.map(tokenize, batched=False)

def collater(data):
	return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], collate_fn=collater)

# Adding accelerator
accelerator = Accelerator(log_with='wandb')
accelerator.init_trackers('trl_accelerate', config=config)
device = accelerator.device
print("DEVICE: ", device)

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

#gpt_blocks = list(gpt2_model.transformer.h)[:-1]
#for m in gpt_blocks:
#	for p in m.parameters():
#		p.requires_grad = False

gen_kwargs = {
	"max_length": 64,
    "min_length": 20,
	"pad_token_id": gpt2_tokenizer.eos_token_id
}

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, device, **config, accelerator=accelerator)

total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

# Prepare accelerator
# fsdp requires model is prepared before optimizer for memory efficiency

gpt2_model, optimizer, dataloader = accelerator.prepare(gpt2_model, ppo_trainer.optimizer, dataloader)
gpt2_model = gpt2_model.to(accelerator.device)
# Fix for running without fsdp or deepspeed
ppo_trainer.optimizer = optimizer

# Run batches to clear errors
rank = torch.distributed.get_rank()
if rank == 0:
	text_in = "The purpose of life is "
elif rank == 1:
	text_in = "Are you human? "

dummy_input = gpt2_tokenizer(text_in, return_tensors="pt").to(accelerator.device)

# had to run this 1 time at the start else was giving device mismatch error.
# So, before directly using `model.generate` pass a batch with dummy data through the model
# outputs = gpt2_model(**dummy_input)
unwrapped_model = accelerator.unwrap_model(gpt2_model)

print("NUM EPOCHS: ", total_ppo_epochs) if accelerator.is_main_process else None
for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader)), disable=not accelerator.is_local_main_process):
	logs, timing = dict(), dict()
	#print('batch', batch)
	t0 = time.time()
	query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
	#### Get response from gpt2
	t = time.time()
	response_tensors = []
	for i in range(config['batch_size']):
		gen_len = output_size()
		with torch.no_grad():
			dummy_outputs = gpt2_model(**dummy_input)
			response = unwrapped_model.generate(query_tensors[i].unsqueeze(dim=0), synced_gpus=True, **gen_kwargs)
		response_tensors.append(response.squeeze()[-gen_len:])

	batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
	timing['time/get_response'] = time.time()-t

	#### Compute sentiment score
	texts = [q + r for q,r in zip(batch['query'], batch['response'])]
	# Ouptut is a list of lists of containing two dictionaries corresponding to pos/neg class and score
	# may be dependent on the hf version
	pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
	rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
	mean_reward = torch.mean(rewards).item()
	accelerator.log({"mean_reward": mean_reward}, step=epoch)

	#### Run PPO step
	#### Get logprobs
	stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

print("FINISHED TRAINING")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
	gpt2_model = accelerator.unwrap_model(gpt2_model)
	#accelerator.save(gpt2_model.state_dict(), 'gpt2-imbd-pos-vs')
	print("FINISHED SAVING MODEL")
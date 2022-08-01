import torch
import wandb
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

config = {
	"model_name": "lvwerra/gpt2-imdb",
	"cls_model_name": "lvwerra/distilbert-imdb",
	"steps": 20000,
	"batch_size": 4,
	"forward_batch_size": 1,
	"ppo_epochs": 1,
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
		return np.random.choice(self.values)

input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])
#input_size = config["txt_in_max_len"]
#output_size = config["txt_out_max_len"

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
accelerator = Accelerator()
device = accelerator.device
print("DEVICE: ", device)

if accelerator.is_main_process:
	wandb.init(name='trl-test', project='trl-test', config=config,)

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['model_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['model_name'])

#gpt_blocks = list(gpt2_model.transformer.h)[:-1]
#for m in gpt_blocks:
#	for p in m.parameters():
#		p.requires_grad = False

if accelerator.is_main_process:
	wandb.watch(gpt2_model, log='all')

#gpt2_model.to(device)
#gpt2_model_ref.to(device)



gen_kwargs = {
	"min_length":-1,
	"top_k": 0.0,
	"top_p": 1.0,
	"do_sample": True,
	"pad_token_id": gpt2_tokenizer.eos_token_id
}

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, device, **config, accelerator=accelerator)

total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

# Prepare accelerator
# fsdp requires model is prepared before optimizer for memory efficiency
gpt2_model = gpt2_model.to(accelerator.device)
gpt2_model = accelerator.prepare(gpt2_model)
optimizer, dataloader = accelerator.prepare(ppo_trainer.optimizer, dataloader)
# Fix for running without fsdp or deepspeed
#gpt2_model = gpt2_model.module
ppo_trainer.optimizer = optimizer

# Run batches to clear errors
rank = torch.distributed.get_rank()
if rank == 0:
	text_in = "The purpose of life is "
elif rank == 1:
	text_in = "Are you human? "

batch = gpt2_tokenizer(text_in, return_tensors="pt").to(accelerator.device)

# had to run this 1 time at the start else was giving device mismatch error.
# So, before directly using `model.generate` pass a batch with dummy data through the model 
outputs = gpt2_model(**batch)
print(batch)

# Use for generation
unwrapped_model = accelerator.unwrap_model(gpt2_model)

print("NUM EPOCHS: ", total_ppo_epochs) if accelerator.is_main_process else None
for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader)), disable=not accelerator.is_local_main_process):
	logs, timing = dict(), dict()
	#print('batch', batch)
	t0 = time.time()
	query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
	#model_input = {"input_ids": torch.tensor([t for t in batch["tokens"]]).to(device), "attention_mask": torch.tensor([t for t in batch["attention_mask"]]).to(device)}
	#### Get response from gpt2
	t = time.time()
	response_tensors = []
	with torch.no_grad():
		for i in range(config['batch_size']):
			#unwrapped_model = accelerator.unwrap_model(gpt2_model)
			gen_len = output_size()
			
			response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
										max_new_tokens=gen_len, **gen_kwargs)
			response_tensors.append(response.squeeze()[-gen_len:])
	#gen_len = output_size
	#response = gpt2_model.generate(**model_input, max_new_tokens=gen_len, **gen_kwargs)
	#batch['response'] = gpt2_tokenizer.batch_decode(response)
	# Form query and response tensors to feed into ppo object
	#response_tensors = response.view((len(batch), -1))
	#query_tensors = model_input["input_ids"].view((len(batch), -1))
	batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
	timing['time/get_response'] = time.time()-t

	#### Compute sentiment score
	t = time.time()
	texts = [q + r for q,r in zip(batch['query'], batch['response'])]
	# Ouptut is a list of lists of containing two dictionaries corresponding to pos/neg class and score
	# may be dependent on the hf version
	pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
	
	rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
	timing['time/get_sentiment_preds'] = time.time()-t

	#### Run PPO step
	t = time.time()
	stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
	timing['time/optimization'] = time.time()-t

	#### Log everything
	timing['time/epoch'] = time.time()-t0
	table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
	if accelerator.is_main_process:
		logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
		logs.update(timing)
		logs.update(stats)
		logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
		logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
		logs['env/reward_dist'] = rewards.cpu().numpy()
		wandb.log(logs)

	#### get a batch from the dataset

accelerator.wait_for_everyone()

bs = 16
game_data = dict()
ds.set_format("pandas")
df_batch = ds[:].sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = df_batch['tokens'].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
	gen_len = output_size()
	output = gpt2_model_ref.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0),
									 max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
	response_tensors_ref.append(output)
	output = gpt2_model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0),
								 max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
	response_tensors.append(output)

#### decode responses
game_data['response (before)'] = [gpt2_tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data['response (after)'] = [gpt2_tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
game_data['rewards (before)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
game_data['rewards (after)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results

print('mean:')
display(df_results[["rewards (before)", "rewards (after)"]].mean())
print()
print('median:')
display(df_results[["rewards (before)", "rewards (after)"]].median())


accelerator.wait_for_everyone()
if accelerator.is_main_process:
	gpt2_model = accelerator.unwrap_model(gpt2_model)
	accelerator.save(gpt2_model.state_dict(), 'gpt2-imbd-pos-vs')
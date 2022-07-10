import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset

from transformers import GPT2Tokenizer, pipeline

from trl.gptneo import GPTNeoHeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

config = {
	"model_name": "EleutherAI/gpt-neo-125M",
	"steps": 20000,
	"batch_size": 64,
	"forward_batch_size": 16,
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)
pipe_device = 0 if torch.cuda.is_available() else -1

wandb.init(name='trl-test-neo', project='trl-test', config=config,)

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

text = 'this movie was really bad!!'
sentiment_pipe(text, **sent_kwargs)

text = 'this movie was really good!!'
sentiment_pipe(text, **sent_kwargs)

#MODIFY MODEL HERE
gptneo_model = GPTNeoHeadWithValueModel.from_pretrained(config['model_name'])
gptneo_model_ref = GPTNeoHeadWithValueModel.from_pretrained(config['model_name'])

gptneo_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
gptneo_tokenizer.pad_token = gptneo_tokenizer.eos_token

wandb.watch(gptneo_model, log='all')

gptneo_model.to(device)
gptneo_model_ref.to(device)

class LengthSampler:
	def __init__(self, min_value, max_value):
		self.values = list(range(min_value, max_value))
	def __call__(self):
		return np.random.choice(self.values)

input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

def tokenize(sample):
	sample["tokens"] = gptneo_tokenizer.encode(sample["review"])[:input_size()]
	sample["query"] = gptneo_tokenizer.decode(sample["tokens"])
	return sample

ds = ds.map(tokenize, batched=False)

gen_kwargs = {
	"min_length":-1,
	"top_k": 0.0,
	"top_p": 1.0,
	"do_sample": True,
	"pad_token_id": gptneo_tokenizer.eos_token_id
}

def collater(data):
	return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], collate_fn=collater)

ppo_trainer = PPOTrainer(gptneo_model, gptneo_model_ref, gptneo_tokenizer, **config)

total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

print("NUM EPOCHS: ", total_ppo_epochs)
for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
	logs, timing = dict(), dict()
	t0 = time.time()
	query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

	#### Get response from gpt2
	t = time.time()
	response_tensors = []
	for i in range(config['batch_size']):
		gen_len = output_size()
		response = gptneo_model.generate(query_tensors[i].unsqueeze(dim=0),
									   max_new_tokens=gen_len, **gen_kwargs)
		response_tensors.append(response.squeeze()[-gen_len:])
	batch['response'] = [gptneo_tokenizer.decode(r.squeeze()) for r in response_tensors]
	timing['time/get_response'] = time.time()-t

	#### Compute sentiment score
	t = time.time()
	texts = [q + r for q,r in zip(batch['query'], batch['response'])]
	pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
	#print(pipe_outputs)
	rewards = torch.tensor([output["score"] if output['label'] == 'POSITIVE' else -output['score'] for output in pipe_outputs]).to(device)
	timing['time/get_sentiment_preds'] = time.time()-t

	#### Run PPO step
	t = time.time()
	stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
	timing['time/optimization'] = time.time()-t

	#### Log everything
	timing['time/epoch'] = time.time()-t0
	table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
	logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
	logs.update(timing)
	logs.update(stats)
	logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
	logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
	logs['env/reward_dist'] = rewards.cpu().numpy()
	wandb.log(logs)

	#### get a batch from the dataset
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
	output = gptneo_model_ref.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
									 max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
	response_tensors_ref.append(output)
	output = gptneo_model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
								 max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
	response_tensors.append(output)

#### decode responses
game_data['response (before)'] = [gptneo_tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data['response (after)'] = [gptneo_tokenizer.decode(response_tensors[i]) for i in range(bs)]

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

gptneo_model.save_pretrained('gptneo-imdb-pos-v2', push_to_hub=False)
gptneo_tokenizer.save_pretrained('gptneo-imdb-pos-v2', push_to_hub=False)
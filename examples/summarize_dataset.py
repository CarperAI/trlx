import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

def get_dataset_from_jsonl(jsonl_file, return_summary=True):
    with open(jsonl_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    post_list = []
    summ_list = []
    for d in dataset:
        if return_summary:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: {d['summary']}"
        else:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: "
            summ_list.append(d['summary'])
        post_list.append(post)
    if return_summary == False:
        return post_list, summ_list
    return post_list
    

class TLDRDataset(Dataset):

  def __init__(self, train_path, tokenizer, max_length=550):

    self.post_list = get_dataset_from_jsonl(train_path)
    if "valid" in train_path:
        self.post_list = self.post_list[0:2000]
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.input_ids = []
    self.attn_masks = []
    
  def __len__(self):
    return len(self.post_list)

  def __getitem__(self, idx):
    txt = self.post_list[idx]
    encodings_dict = self.tokenizer(
        txt, truncation=True, 
        max_length=self.max_length, 
        padding="max_length"
    )
    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_masks = torch.tensor(encodings_dict['attention_mask'])
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attn_masks,
        "labels": input_ids
    }
    
       
    
class ComparisionDataset(Dataset):

    def __init__(self, comparision_path, tokenizer, max_length=550):
        with open(comparision_path, 'r') as f:
            dataset = [json.loads(line) for line in f]
        
        if 'valid' in comparision_path:
            dataset = dataset[:2000]
            
        self.tokenizer = tokenizer
        self.lst_post = []
        self.lst_summaries_0 = []
        self.lst_summaries_1 = []
        self.labels = []
        self.max_length = max_length
        count_zero = 0
        count_one = 0
        
        def make_text(post, summarize):
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        for sample in dataset:
            self.lst_post.append(sample['info']['post'])
            if sample['choice'] == 0:
                self.lst_summaries_0.append(make_text(sample['info'], sample['summaries'][0]['text']))
                self.lst_summaries_1.append(make_text(sample['info'], sample['summaries'][1]['text']))
                count_zero += 1
            else:
                self.lst_summaries_0.append(make_text(sample['info'], sample['summaries'][1]['text']))
                self.lst_summaries_1.append(make_text(sample['info'], sample['summaries'][0]['text']))
                count_one += 1
            self.labels.append(0)
        print("count_zero", count_zero)
        print("count_one", count_one)


    def __len__(self):
        return len(self.lst_post)
    
    def __getitem__(self, idx):
        summ0 = self.lst_summaries_0[idx]
        summ1 = self.lst_summaries_1[idx]
        choice = self.labels[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1], 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length"
        )
        input_ids = torch.tensor(encodings_dict['input_ids'])
        attention_mask = torch.tensor(encodings_dict['attention_mask'])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

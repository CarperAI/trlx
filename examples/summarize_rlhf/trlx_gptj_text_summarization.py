import torch
from transformers import  AutoTokenizer
from reward_model.reward_model import GPTRewardModel
import trlx
from trlx.data.configs import TRLConfig
from typing import List
from datasets import load_dataset



REWARD_CHECKPOINT_PATH = "/admin/home-duyphung/refactor_summarize_rlhf/trlx/examples/summarize_rlhf/reward_model_inspect/ckpts/openai_comparison_summary/gpt-j/checkpoint-1700/pytorch_model.bin"

if __name__ == "__main__":
    
    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel("pvduy/openai_summarize_sft_gptj")
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))
    rw_model.to(rw_device)
    
    def reward_fn(samples: List[str]):
        
        original_samples = [text.split('TL;DR:')[0] + 'TL;DR: ' for text in samples]
        original_samples = [text + train_post_summ[text.strip()] for text in original_samples]
        
        ori_lst_scores = []
        batch_size = 2
        for i in range(0, len(original_samples), batch_size):
            sub_samples = original_samples[i:i+batch_size]
            sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                    sub_samples, 
                    truncation=True, 
                    max_length=550, 
                    padding="max_length",
                    return_tensors="pt"
            )
            input_ids = encodings_dict['input_ids'].to(rw_device)
            attn_masks = encodings_dict['attention_mask'].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            ori_lst_scores.append(sub_scores['chosen_end_scores'])
        ori_scores = torch.cat(ori_lst_scores, dim=0)
        
        lst_scores = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i:i+batch_size]
            sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                    sub_samples, 
                    truncation=True, 
                    max_length=550, 
                    padding="max_length",
                    return_tensors="pt"
            )
            input_ids = encodings_dict['input_ids'].to(rw_device)
            attn_masks = encodings_dict['attention_mask'].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            lst_scores.append(sub_scores['chosen_end_scores'])
        scores = torch.cat(lst_scores, dim=0)
        norms_scores = scores  - ori_scores
        return norms_scores

    config = TRLConfig.load_yaml("configs/ppo_config_summ_gptj.yml")
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    
    train_openai_summ = [
        sample['prompt'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="train_rl")
    ]
    train_labels = [
        sample['label'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="train_rl")
    ]
    
    val_openai_summ = [
        sample['prompt'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="valid")
    ]
    val_labels = [
        sample['label'] for sample in load_dataset("pvduy/openai_summarize_tldr", split="valid")
    ]    
    
    train_post_summ = {}
    for i in range(len(train_openai_summ)):
       tmp = tokenizer.decode(
                tokenizer(
                    train_openai_summ[i],
                    truncation=True,
                    max_length=max_length
                )['input_ids'],
                skip_special_tokens=True, 
            ).strip()
       train_post_summ[tmp] = train_labels[i]
    
    for i in range(len(val_openai_summ)):
        tmp = tokenizer.decode(
                tokenizer(
                    val_openai_summ[i],
                    truncation=True,
                    max_length=max_length
                )['input_ids'],
                skip_special_tokens=True, 
            ).strip()
        train_post_summ[tmp] = val_labels[i]

    
    prompts = train_openai_summ
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_openai_summ[0:1000],
        config=config
    )

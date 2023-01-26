# Trains a model to encode and decode a number in a poem.

import os
import yaml

import yaml
import trlx
from typing import List
from trlx.data.configs import TRLConfig
import transformers
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

default_config = yaml.safe_load(open("configs/ppo_config.yml"))

# get gpt2 tokenizer
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

def get_last_digit(sample: List[str]) -> int:
    """
    Extract last char from a sample, check if it's a digit, otherwise return -1
    """
    last_word = sample[-1]
    if last_word.isdigit():
        return int(last_word)
    else:
        return -1

def reward_fn(trajectories: List[List]) -> List[float]:
    """
    Inputs have the form [digit, sample_1, sample_2]
    Return if the last digit of sample_2 is the same as the digit
    """
    for sample in trajectories:
        assert len(sample) == 3
    reconstructed_digits = list(map(get_last_digit, trajectories[:, 2]))
    return [1 if digit == reconstructed_digit else 0 for digit, reconstructed_digit in zip(trajectories[:, 0], reconstructed_digits)]


def make_tokens(text, batch_size, device) -> BatchEncoding:
    tokens = tokenizer(text, return_tensors="pt").to(device)
    tokens.input_ids = tokens.input_ids.repeat(batch_size, 1)
    tokens.attention_mask = tokens.attention_mask.repeat(batch_size, 1)
    tokens.labels = tokens.input_ids.clone()
    return tokens

def concat_tokens(input_ids_1, attn_mask_1, input_ids_2, attn_mask_2) -> BatchEncoding:
    """
    Concatenate two sequences of tokens, moving the padding to the end
    """
    batch_size = input_ids_1.shape[0]
    input_ids = torch.zeros((batch_size, input_ids_1.shape[1] + input_ids_2.shape[1]), dtype=torch.long, device=input_ids_1.device)
    attn_mask = torch.zeros((batch_size, attn_mask_1.shape[1] + attn_mask_2.shape[1]), dtype=torch.long, device=attn_mask_1.device)
    for i in range(batch_size):
        input_ids[i, :] = torch.cat((
            input_ids_1[i, attn_mask_1[i, :] == 1],
            input_ids_2[i, attn_mask_2[i, :] == 1],
            input_ids_1[i, attn_mask_1[i, :] == 0],
            input_ids_2[i, attn_mask_2[i, :] == 0],
        ))
        attn_mask[i, :] = torch.cat((
            attn_mask_1[i, attn_mask_1[i, :] == 1],
            attn_mask_2[i, attn_mask_2[i, :] == 1],
            attn_mask_1[i, attn_mask_1[i, :] == 0],
            attn_mask_2[i, attn_mask_2[i, :] == 0],
        ))
    # make a tokens object
    tokens = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": input_ids.clone(),
    }

def encoder_decoder_experience_fn(model, batch):
    """
    :model: AccelerateRLModel
        (has model.generate_and_calc_logprobs, has model.tokenizer)
    :batch: an object that has .input_ids, .attention_mask, .labels
    
    :return: trajectories, 
             data={'samples': samples, 'all_logprobs': all_logprobs, 'all_ref_logprobs': all_ref_logprobs, 'query_tensors': query_tensors, 'response_tensors': response_tensors, 'all_values': all_values}, 
             stats
            
    Use model.generate_and_calc_logprobs to return all data needed for PPO for a complex trajectory.
    model.generate_and_calc_logprobs : (batch) 
       -> {'samples': samples, 'all_logprobs': all_logprobs, 'all_ref_logprobs': all_ref_logprobs, 'query_tensors': query_tensors, 'response_tensors': response_tensors, 'all_values': all_values}
    The trajectory for each poem is as follows:
    Sample a digit from {0, 1}.
    First run:
    f"Fact: x = {digit}
    Continue the poem:\n
    {poem}
    The"
    --> poem_continuation
    Second run:
    f"{poem}
    The{poem_continuation}\n
    Recall fact: x is either 0 or 1.
    Answer: x ="
    --> answer
    Everything needs to be tokenized all the time.
    """

    # batch is an object that has .input_ids, .attention_mask, .labels; for example
    # batch {'input_ids': tensor([[ 3], [ 8]]), 'attention_mask': tensor([[1], [1]]), 'labels': tensor([[ 8], [10]])}

    batch_size = batch.input_ids.shape[0]
    device = batch.input_ids.device
    digits = [0, 1]


    for digit in digits:
        fact_tokens = make_tokens(f"Fact: x = {digit}\nContinue the poem:\n\n", batch_size, device)
        the_tokens = make_tokens("\nThe", batch_size, device)
        recall_tokens = make_tokens(f"\nRecall fact: x is either 0 or 1.\nAnswer: x =", batch_size, device)

        # First run
        first_run_inputs = concat_tokens(fact_tokens.input_ids, fact_tokens.attention_mask, batch.input_ids, batch.attention_mask)
        first_run_inputs = concat_tokens(first_run_inputs[0], first_run_inputs[1], the_tokens.input_ids, the_tokens.attention_mask)








``






def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    train_path = "examples/poems/poetry_big_train_qa.csv"
    data = pd.read_csv(train_path)
    prompts = data["question"].tolist() # everthing else goes in the trajectory function

    model = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        config=config,
    )


if __name__ == "__main__":
    main()

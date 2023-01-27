# Trains a model to encode and decode a number in a poem.

import os
import yaml

import yaml
import trlx
from typing import List
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import RunElementBatch
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.data.accelerate_base_datatypes import PromptBatch
import transformers
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import torch
import pandas as pd

default_config = yaml.safe_load(open("examples/experience_demo/configs/ppo_config.yml"))

def get_last_digit(sample: List[str]) -> int:
    """
    Extract last char from a sample, check if it's a digit, otherwise return -1
    """
    if len(sample) == 0:
        return -2
    last_word = sample[-1]
    if last_word.isdigit():
        return int(last_word)
    else:
        return -1

def reward_fn(trajectories: List[List]) -> List[float]:
    """
    trajectories is a list of lists having the form [digit, prompt_1, output_1, prompt_2, output_2]
    Return if the last digit of output_2 is the same as the digit
    """
    for sample in trajectories:
        assert len(sample) == 5
    #reconstructed_digits = list(map(get_last_digit, trajectories[:, 2]))
    # can't do that with list of lists
    reconstructed_digits = list(map(get_last_digit, [sample[4] for sample in trajectories]))
    #return [1 if digit == reconstructed_digit else 0 for digit, reconstructed_digit in zip(trajectories[:, 0], reconstructed_digits)]
    reward_list = [1 if digit == reconstructed_digit else 0 for digit, reconstructed_digit in zip([sample[0] for sample in trajectories], reconstructed_digits)]
    print(f"reward_mean = {np.mean(reward_list)}")
    return reward_list



def encoder_decoder_experience_fn(trainer, batch):
    """
    :trainer: AccelerateRLTrainer
        (has trainer.orch.generate_and_calc_logprobs, which returns data : RunElementBatch and stats : dict)
        (has trainer.decode)
    :batch: an object that has .input_ids, .attention_mask, .labels
    
    :return: trajectories (the input type to reward_fn), 
             data : RunElementBatch
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
    """

    # batch is an object that has .input_ids, .attention_mask, .labels; for example
    # batch {'input_ids': tensor([[ 3], [ 8]]), 'attention_mask': tensor([[1], [1]]), 'labels': tensor([[ 8], [10]])}

    batch_size = batch.input_ids.shape[0]
    print(f"batch_size = {batch_size}")
    device = batch.input_ids.device
    query_tensors = batch.input_ids
    digits = list(np.random.randint(0, 2, batch_size)) # sample a digit from {0, 1}

    # The key architectural constraint is that alll trainer.orch.generate_and_calc_logprobs should be parallel over the batch
    # Do everything in string space 
    fact_strs = [f"Fact: x = {digits[i]}\nContinue the poem:\n\n" for i in range(batch_size)]
    recall_str = f"\nRecall fact: x is either 0 or 1.\nAnswer: x ="

    # Detokenize the text
    _, str_prompts, _ = trainer.decode(
        query_tensors, query_tensors, # this is a hack to get the text, we are doing redundant tokenization
    )

    first_run_strs = [""] * batch_size
    # First run
    for i in range(batch_size):
        first_run_strs[i] = fact_strs[i] + str_prompts[i] + "\nThe"

    # Encode the first run
    #first_run_batch = trainer.tokenizer(first_run_strs)
    # but tensors
    first_run_batch = trainer.tokenizer(first_run_strs, return_tensors="pt", padding=True, truncation=True)

    # Generate the first run
    #import code; print("first_run_data"); code.interact(local=locals())
    first_run_data, first_run_stats = trainer.orch.generate_and_calc_logprobs(first_run_batch)

    first_run_str_prompts = first_run_data['str_prompts']
    first_run_str_outputs = first_run_data['str_outputs']

    # Second run
    second_run_strs = [""] * batch_size
    for i in range(batch_size):
        second_run_strs[i] = str_prompts[i] + "\nThe" + first_run_str_outputs[i] + recall_str

    # Encode the second run
    second_run_batch = trainer.tokenizer(second_run_strs, return_tensors="pt", padding=True, truncation=True)

    # Generate the second run
    second_run_data, second_run_stats = trainer.orch.generate_and_calc_logprobs(second_run_batch, max_new_tokens=1)

    # Decode the second run
    second_run_str_prompts = second_run_data['str_prompts']
    second_run_str_outputs = second_run_data['str_outputs']

    print("AAAAAAAAAAAAAAAAAa")
    if digits[0] == 0 and digits[1] == 0 and digits[2] == 0 and digits[3] == 0: # prob 1/16
        print("first_run_str_prompts\n", first_run_str_prompts)
        print("first_run_str_outputs\n", first_run_str_outputs)
        print("second_run_str_prompts\n", second_run_str_prompts)
        print("second_run_str_outputs\n", second_run_str_outputs)
    #import code; print("just before data concatenation"); code.interact(local=locals())

    # RunElementBatch has an __add__ method which should do the right thing
    data = first_run_data + second_run_data
    # sum up a dict over keys 
    stats = {k: first_run_stats[k] + second_run_stats[k] for k in first_run_stats}

    trajectories = [] # list of lists, of the form [digit, prompt_1, output_1, prompt_2, output_2]
    # convert to a list of lists
    for i in range(batch_size):
        trajectories.append([digits[i], first_run_str_prompts[i], first_run_str_outputs[i], second_run_str_prompts[i], second_run_str_outputs[i]])

    return trajectories, data, stats





def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    train_path = "examples/experience_demo/poems/poetry_big_train_qa.csv"
    data = pd.read_csv(train_path)
    prompts = data["question"].tolist() # everthing else goes in the trajectory function

    np.random.seed(42)
    trlx.train(
        reward_fn=reward_fn,
        experience_fn=encoder_decoder_experience_fn,
        prompts=prompts,
        config=config,
    )


if __name__ == "__main__":
    main()

# Trains a model to increment a number by 2, in two steps.

import os
import yaml
import itertools
from typing import List

import trlx
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import RunElementBatch
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.data.accelerate_base_datatypes import PromptBatch
import transformers
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
import torch
import pandas as pd

from examples.experience_demo.utils import AllowListLogitsProcessorList

import wandb

default_config = yaml.safe_load(open("examples/experience_demo/configs/ppo_config_increment.yml"))

def get_last_digit(sample: str) -> int:
    sample = sample.lstrip()
    if len(sample) == 0:
        return -2
    if len(sample) > 1:
        return -3
    last_word = sample[-1]
    if last_word.isdigit():
        return int(last_word)
    else:
        return -1

def double_increment_score(digit : int, intermediate : int, reconstructed_digit : int) -> float:
    if intermediate < 0 or reconstructed_digit < 0:
        return -10
    if digit + 2 == reconstructed_digit:
        return 1
    else:
        return -1

def increment_help_score(digit : int, intermediate : int, reconstructed_digit : int) -> float:
    score_log_file = "examples/experience_demo/increment_log.txt"
    with open(score_log_file, "a") as f:
        f.write(f"{digit} {intermediate} {reconstructed_digit}\n")

    if intermediate < 0 or reconstructed_digit < 0:
        return -10
    if digit + 2 == reconstructed_digit:
        return 2
    elif digit + 1 == intermediate:
        return 1
    else:
        return -1
    
def fix_score(digit : int, intermediate : int, reconstructed_digit : int) -> float:
    if intermediate < 0 or reconstructed_digit < 0:
        return -10
    if reconstructed_digit == 2:
        return 10
    else:
        return 0

def reward_fn(trajectories: List[List]) -> List[float]:
    """
    trajectories is a list of lists having the form [digit, prompt_1, output_1, prompt_2, output_2]
    Pass the digit, intermediate, and reconstructed digit to the score function.
    """
    score = double_increment_score

    for sample in trajectories:
        assert len(sample) == 5
    digits = [sample[0] for sample in trajectories]
    reconstructed_digits = list(map(get_last_digit, [sample[4] for sample in trajectories]))
    intermediate_digits = list(map(get_last_digit, [sample[2] for sample in trajectories]))
    reward_list = [score(digit, intermediate, reconstructed_digit) for digit, intermediate, reconstructed_digit in zip(digits, intermediate_digits, reconstructed_digits)]
    print(f"reward_mean = {np.mean(reward_list)}")
    return reward_list


def prompt_example(input : str) -> str:
    return f"""0
next=1

1
next=2

2
next=3

3
next=4

4
next=5

5
next=6

6
next=7

7
next=8

{input}
next="""


def prompt_tell(input : str) -> str:
    return f"""The next number after {input} is x. x="""

def prompt_think(input : str) -> str:
    return f"""f is a function. f(f(x))=x+2. f({input})="""

def prompt_arrow(input : str) -> str:
    return f"""{input}->"""

logits_processor = AllowListLogitsProcessorList()

def increment_experience_fn(trainer, batch):
    """
    :trainer: AccelerateRLTrainer
        (has trainer.orch.generate_and_calc_logprobs, which returns data : RunElementBatch and stats : dict)
        (has trainer.decode)
    :batch: an object that has .input_ids, .attention_mask, .labels
    
    :return: trajectories (the input type to reward_fn), 
             data : RunElementBatch
             stats
            
    Use model.generate_and_calc_logprobs to return all data needed for PPO for a complex trajectory.
    We completely ignore the dataset (the query tensors).
    
    We use max_new_tokens=1 in all calls to generate_and_calc_logprobs.
    """

    # batch is an object that has .input_ids, .attention_mask, .labels; for example
    # batch {'input_ids': tensor([[ 3], [ 8]]), 'attention_mask': tensor([[1], [1]]), 'labels': tensor([[ 8], [10]])}

    prompt = prompt_arrow

    batch_size = batch.input_ids.shape[0]
    print(f"\nbatch_size = {batch_size}")
    device = batch.input_ids.device
    digits = list(np.random.randint(0, 8, batch_size))

    # The key architectural constraint is that all trainer.orch.generate_and_calc_logprobs should be parallel over the batch
    # Do everything in string space 

    # First run
    first_run_strs = [""] * batch_size
    for i in range(batch_size):
        first_run_strs[i] = prompt(digits[i])

    # Encode the first run
    first_run_batch = trainer.tokenizer(first_run_strs, return_tensors="pt", padding=True, truncation=True)

    # Generate the first run
    #import code; print("first_run_data"); code.interact(local=locals())
    first_run_data, first_run_stats = trainer.orch.generate_and_calc_logprobs(
        first_run_batch, logits_processor=logits_processor,
        max_new_tokens=1,
    )

    first_run_str_prompts = list(itertools.chain.from_iterable(first_run_data['str_prompts']))
    first_run_str_outputs = list(itertools.chain.from_iterable(first_run_data['str_outputs']))

    # Second run
    second_run_strs = [""] * batch_size
    for i in range(batch_size):
        second_run_strs[i] = prompt(first_run_str_outputs[i].replace(' ', ''))

    # Encode the second run
    second_run_batch = trainer.tokenizer(second_run_strs, return_tensors="pt", padding=True, truncation=True)

    # Generate the second run
    second_run_data, second_run_stats = trainer.orch.generate_and_calc_logprobs(
        second_run_batch, logits_processor=logits_processor,
        max_new_tokens=1,
    )

    # Decode the second run
    second_run_str_prompts = list(itertools.chain.from_iterable(second_run_data['str_prompts']))
    second_run_str_outputs = list(itertools.chain.from_iterable(second_run_data['str_outputs']))

    print("first_run_str_prompts\n", first_run_str_prompts[:4])
    print("first_run_str_outputs\n", first_run_str_outputs[:4])
    print("second_run_str_prompts\n", second_run_str_prompts[:4])
    print("second_run_str_outputs\n", second_run_str_outputs[:4])
    #import code; print("just before data concatenation"); code.interact(local=locals())


    # RunElementBatch has an __add__ method which should do the right thing
    datas : List[RunElementBatch] = [first_run_data, second_run_data]
    #import code; print("data"); code.interact(local=locals())
    # sum up a dict over keys 
    stats = {k: first_run_stats[k] + second_run_stats[k] for k in first_run_stats}

    trajectories = [] # list of lists, of the form [digit, prompt_1, output_1, prompt_2, output_2]
    for i in range(batch_size):
        trajectories.append([digits[i], first_run_str_prompts[i], first_run_str_outputs[i], second_run_str_prompts[i], second_run_str_outputs[i]])

    options = {}
    return trajectories, datas, stats, options





def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    # delete score_log_file if it exists
    score_log_file = "examples/experience_demo/increment_log.txt"
    if os.path.exists(score_log_file):
        os.remove(score_log_file)
    train_path = "examples/experience_demo/poems/poetry_big_train_qa.csv"
    data = pd.read_csv(train_path)
    prompts = data["question"].tolist() # we don't use this, we just need a list of strings

    np.random.seed(43)
    trlx.train(
        reward_fn=reward_fn,
        experience_fn=increment_experience_fn,
        prompts=prompts, 
        config=config,
    )


if __name__ == "__main__":
    main()

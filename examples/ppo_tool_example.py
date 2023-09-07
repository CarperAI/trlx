# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from datasets import load_dataset
from trlx.environment.base_tool import ToolEnvironment
from transformers import load_tool

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ppo_init_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            minibatch_size=2,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
        ),
        model=ModelConfig(model_path="codellama/CodeLlama-7b-Instruct-hf", num_layers_unfrozen=8),
        tokenizer=TokenizerConfig(tokenizer_path="codellama/CodeLlama-7b-Instruct-hf", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-7)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            num_value_layers_unfrozen=4,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=256,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

import torch


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    for response, answer in zip(responses, answers):
        reward = 0.0
        if "Error:" in response:
            reward = -1.0
        else:
            response = response.strip()
            answer = answer.strip()
            reward += float(response == answer)
        rewards.append(reward)
    print("rewards: ", rewards)
    print("responses: ", responses)
    print("answers: ", answers)
    return rewards


def create_reward_function(prompt):
    tool_env = ToolEnvironment([load_tool("lvwerra/python-interpreter")], prompt, exact_match_reward)

    def reward_fn(samples, prompts, original_output, **kwargs):
       # for sample in samples:
       #     print("sample: ", sample)
       #     print("======================================")
        rewards = tool_env.get_reward(samples, **{"answers": original_output})
        print("rewards: ", rewards)
        return rewards

    return reward_fn


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(ppo_init_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    ds = load_dataset("gsm8k", "main", split="train")
    ds = ds.rename_columns({"question": "query"})
    ds = ds.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    ds = ds.select(range(1, len(ds)))  # skip the first sample which is used in prompt

    ds_test = load_dataset("gsm8k", "main", split="test").select(range(1, 100))
    ds_test = ds_test.rename_columns({"question": "query"})
    ds_test = ds_test.map(lambda x: {"answer": x["answer"].split("#### ")[1]})

    df = ds.to_pandas()
    df_test = ds_test.to_pandas()

    few_shot_prompt = """\
Example of using a Python API to solve math questions. Make sure that you print the result of your solution.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

<request><PythonInterpreter>
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
print(solution())
<call>72<response>

Result = 72 <submit>

"""

    reward_fn = create_reward_function(few_shot_prompt)

    generate_prompt = """\
{few_shot_prompt}
Q: {query}

<request><PythonInterpreter>"""

    df["query"] = df["query"].apply(
        lambda x: generate_prompt.format(few_shot_prompt=few_shot_prompt, query=x)
    )
    df_test["query"] = df_test["query"].apply(
        lambda x: generate_prompt.format(few_shot_prompt=few_shot_prompt, query=x)
    )

    train_prompts = [
        {"prompt": query, "original_output": answer}
        for (query, answer) in zip(df["query"].tolist(), df["answer"].tolist())
    ]
    eval_prompts = [
        {"prompt": query, "original_output": answer}
        for (query, answer) in zip(df_test["query"].tolist(), df_test["answer"].tolist())
    ]
    trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=config,
        stop_sequences=["<call>", "<endoftext>"],
    )


if __name__ == "__main__":
    main()

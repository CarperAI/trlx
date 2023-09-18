# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline
import numpy as np

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
import pandas as pd
from trlx.environment.base_tool import ToolEnvironment
from transformers import load_tool
from sklearn.model_selection import train_test_split

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ppo_init_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=256,
            epochs=100,
            total_steps=1000,
            batch_size=8,
            checkpoint_interval=10000,
            eval_interval=10,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=1.0e-5)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-6)),
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
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=32,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


def generate_data(n):
    """Generate random arithmetic tasks and answers."""
    tasks, answers = [], []
    for _ in range(n):
        a = np.random.randint(0, 50)
        b = np.random.randint(0, 50)
        op = np.random.choice(["-", "+", "*"])
        tasks.append(f"\n\nWhat is {a} {op} {b}?")
        if op == "-":
            answers.append(a - b)
        elif op == "+":
            answers.append(a + b)
        else:
            answers.append(a * b)
    return tasks, answers


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    responses = [str(response) for response in responses]
    answers = [str(answer) for answer in answers]
    for response, answer in zip(responses, answers):
        reward = 0.0
        if "Error:" in response:
            reward = -1.0
        else:
            response = response.strip()
            answer = answer.strip()
            try:
                response = float(response)
                answer = float(answer)
                if np.abs(response - answer) < 1e-5:
                    reward = 1.0
            except:
                reward = 0
        rewards.append(reward)
    return rewards


def create_reward_function(prompt):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tool_env = ToolEnvironment(
        {"SimpleCalculatorTool": load_tool("ybelkada/simple-calculator")}, prompt, exact_match_reward, tokenizer
    )

    def reward_fn(samples, prompts, original_output, **kwargs):
        rewards = tool_env.get_reward(samples, **{"answers": original_output})
        return rewards

    return reward_fn


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(ppo_init_config().to_dict(), hparams)

    tasks, answers = generate_data(256 * 100)
    tasks = [x.strip("\n") for x in tasks]
    df = pd.DataFrame({"query": tasks, "answer": answers})

    df, df_test = train_test_split(df, test_size=500)
    few_shot_prompt = """\
Q: What is 13 - 3?

<request><SimpleCalculatorTool>13-3<call>10.0<response>

Result=10<submit>

Q: What is 4 * 3?

<request><SimpleCalculatorTool>4*3<call>12.0<response>

Result=12<submit>

Q: What is 1 + 2?

<request><SimpleCalculatorTool>1+2<call>3<response>

Result=3<submit>"""

    reward_fn = create_reward_function(few_shot_prompt)

    generate_prompt = """\
{few_shot_prompt}

Q: {query}

<request><SimpleCalculatorTool>"""

    df["query"] = df["query"].apply(lambda x: generate_prompt.format(few_shot_prompt=few_shot_prompt, query=x))
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

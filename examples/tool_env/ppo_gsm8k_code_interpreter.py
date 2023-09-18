# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import os

from datasets import load_dataset
from transformers import load_tool

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
from trlx.environment.base_tool import ToolEnvironment

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ppo_init_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=768,
            epochs=100,
            total_steps=1000,
            batch_size=32,
            minibatch_size=1,
            checkpoint_interval=10000,
            eval_interval=20,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=True,
            checkpoint_dir="/fsx/home-duyphung/trlx_checkpoints",
        ),
        model=ModelConfig(model_path="codellama/CodeLlama-7b-Instruct-hf", num_layers_unfrozen=12),
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
            num_value_layers_unfrozen=8,
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
    return rewards


def create_reward_function(prompt):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    tool_env = ToolEnvironment([load_tool("lvwerra/python-interpreter")], prompt, exact_match_reward, tokenizer)

    def reward_fn(samples, prompts, original_output, **kwargs):
        rewards = tool_env.get_reward(samples, **{"answers": original_output})
        return rewards

    return reward_fn


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(ppo_init_config().to_dict(), hparams)

    ds = load_dataset("gsm8k", "main", split="train")
    ds = ds.rename_columns({"question": "query"})
    ds = ds.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    ds = ds.select(range(1, len(ds)))  # skip the first sample which is used in prompt

    ds_test = load_dataset("gsm8k", "main", split="test").select(range(1, 500))
    ds_test = ds_test.rename_columns({"question": "query"})
    ds_test = ds_test.map(lambda x: {"answer": x["answer"].split("#### ")[1]})

    df = ds.to_pandas()
    df_test = ds_test.to_pandas()

    few_shot_prompt = """\
Instruction: Using a Python API to solve math questions.\
Write function solution to solve the following questions, then "print(solution())" to output the result.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

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

Question: Michael loves to paint and sells his creations. He charges $100 for a large painting and $80 for a small painting.\
At his last art show, he sold 5 large paintings and 8 small paintings. How much did he earn in all?

<request><PythonInterpreter>
def solution():
    price_large = 100
    price_small = 80
    paintings_large = 5
    paintings_small = 8
    total_large_price = price_large * paintings_large
    total_small_price = price_small * paintings_small
    total = total_large_price + total_small_price
    result = total
    return result
print(solution())
<call>1140<response>

Result = 1140 <submit>

"""

    reward_fn = create_reward_function(few_shot_prompt)

    generate_prompt = """\
{few_shot_prompt}
Question: {query}

<request><PythonInterpreter>"""

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

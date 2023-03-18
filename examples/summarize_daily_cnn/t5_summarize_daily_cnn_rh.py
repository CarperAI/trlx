from typing import List
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages" "by running `pip install evaluate`"
    )

config = TRLConfig(
    train=TrainConfig(
        seq_length=612,
        epochs=100,
        total_steps=10000,
        batch_size=1,
        checkpoint_interval=50000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path="/data/models/3b-checkpoint-12033",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="/data/models/3b-checkpoint-12033",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=12,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 100,
        },
        gen_experience_kwargs={
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)

def _truncate_text_to_num_tokens(text, max_length, tokenizer):
    length = sum([len(t) for t in tokenizer.tokenize(text)[:max_length]])
    return text[:length]

if __name__ == "__main__":

#     def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
#         original_summaries = [prompt_label[prompt.strip()] for prompt in prompts]
#         scores = [
#             meteor.compute(predictions=[output.strip()], references=[original])["meteor"]
#             for (original, output) in zip(original_summaries, outputs)
#         ]
#         return scores
    # just try reward fn of negative output length
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        scores = [-1*len(output) for output in outputs]
        return scores

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    
    dataset = pd.read_csv('/data/data/outputs-75614-10k.csv')
    prompts = []
    val_prompts = []
    for i, r in dataset.iterrows():
        if i < 8000:
            prompts.append(_truncate_text_to_num_tokens(r.text, max_length, tokenizer))
        else:
            val_prompts.append(_truncate_text_to_num_tokens(r.text, max_length, tokenizer))

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )

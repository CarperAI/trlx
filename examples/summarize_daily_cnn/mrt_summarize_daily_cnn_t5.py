from typing import List

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
from trlx.models.modeling_mrt import MRTConfig

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
        total_steps=100000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateMRTTrainer",
        tracker=None,
    ),
    model=ModelConfig(
        model_path="google/flan-t5-small",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="google/flan-t5-small", #### change to reasonable value
        truncation_side="right", # what is this?
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
    method=MRTConfig(
        name="MRTConfig",
        # n_updates_per_batch=1, #### MRT
        num_rollouts=512,
        chunk_size=4,
        ppo_epochs=4,
        # init_kl_coef=0.05,
        # target=6,
        # horizon=10000,
        # gamma=0.99,
        # lam=0.95,
        # cliprange=0.2,
        # cliprange_value=0.2,
        # vf_coef=1.0,
        num_candidates=16,
        ce_loss_weight=0.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={ # for evaluation
            "max_new_tokens": 100,
            # TODO: what should the defaults here be
        },
        gen_experience_kwargs={ # for rollouts
            "max_new_tokens": 100,
            "num_beams": 16, # should be same as nb_candidates
            "num_return_sequences": 16, # should be same as nb_candidates
            "do_sample": False,
            "temperature": 1.0,
            # "top_k": 50,
            # "top_p": 0.95,
        },
    ),
)

    # gen_kwargs = {
    #     "min_length":-1,
    #     "top_k": config['top_k'],
    #     "top_p": 1.0,
    #     "temperature": config["temperature"],
    #     "do_sample": config['do_sample'],
    #     "num_beams": config['num_beams'],
    #     "max_length": config['max_length'],
    #     # "pad_token_id": model.eos_token_id,
    #     "num_return_sequences": config['candidate_size'],
    # }
    # eval_kwargs = {
    #     # "early_stopping": True,
    #     # "length_penalty": 2.0,
    #     "min_length":-1,
    #     "top_k": 0.0,
    #     # "top_p": 1.0,
    #     "do_sample": False,
    #     "num_beams": config['eval_num_beams'],
    #     # "no_repeat_ngram_size": 3,
    #     "max_length": config['max_length'],
    # }


meteor = evaluate.load("meteor")  # use meteor as the reward function

if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        original_summaries = [prompt_label[prompt.strip()] for prompt in prompts]
        scores = [
            meteor.compute(predictions=[output.strip()], references=[original])["meteor"]
            for (original, output) in zip(original_summaries, outputs)
        ]
        return scores

    dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="data")

    # take 20,000 samples from the training set as prompts for training
    prompts = dataset["train"]["article"][0:1200]
    summaries = dataset["train"]["highlights"][0:1200]
    prompts = ["Summarize: " + prompt for prompt in prompts]

    # take 1,000 samples from the validation set as prompts for evaluation
    val_prompts = ["Summarize: " + prompt for prompt in dataset["validation"]["article"][0:1000]]
    val_summaries = dataset["validation"]["highlights"][0:1000]

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    for i in tqdm(range(len(prompts))):
        key = tokenizer.decode(
            tokenizer(prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = summaries[i]

    for i in tqdm(range(len(val_prompts))):
        key = tokenizer.decode(
            tokenizer(val_prompts[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = val_summaries[i]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )

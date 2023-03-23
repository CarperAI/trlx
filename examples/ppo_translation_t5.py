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
        total_steps=100000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        tracker="wandb"
        # tracker=None
    ),
    model=ModelConfig(
        model_path="google/flan-t5-large",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="google/flan-t5-large",
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


comet_metric = evaluate.load("comet", "wmt20-comet-da")
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")


if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        # WHAT should samples be for translation?
        original_sents = [translation_map[prompt.strip()] for prompt in prompts]

        scores = comet_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
            sources=[original["src"] for original in original_sents],
        )["scores"]
        return scores

    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        # Compute BLEU and CHRF
        original_sents = [translation_map[prompt.strip()] for prompt in prompts]

        comet_score = comet_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
            sources=[original["src"] for original in original_sents],
        )['mean_score']

        bleu_score = bleu_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
        )['bleu']

        chrf_score = bleu_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
        )['score']

        return {
            'bleu': bleu_score,
            'chrf': chrf_score,
            'comet': comet_score
        }

    train_dataset = load_dataset(
        "wmt16", "de-en", split="train", cache_dir="/home/aiscuser/dev/trlx_mrt/examples/notebooks/data", streaming=True
    )
    valid_dataset = load_dataset(
        "wmt16",
        "de-en",
        split="validation",
        cache_dir="/home/aiscuser/dev/trlx_mrt/examples/notebooks/data",
        streaming=False,
    )

    src_lang = "en"
    tgt_lang = "de"
    PREFIX = "translate English to German: "

    # take 20,000 samples from the training set as prompts for training
    original_src_dataset = [sent_pair["translation"][src_lang] for sent_pair in train_dataset.take(20000)]
    tgt_dataset = [sent_pair["translation"][tgt_lang] for sent_pair in train_dataset.take(20000)]
    src_dataset = [PREFIX + src_sent for src_sent in original_src_dataset]

    # take 1,000 samples from the validation set as prompts for evaluation
    val_original_src_dataset = [sent_pair[src_lang] for sent_pair in valid_dataset["translation"][0:1000]]
    val_tgt_dataset = [sent_pair[tgt_lang] for sent_pair in valid_dataset["translation"][0:1000]]
    val_src_dataset = [PREFIX + src_sent for src_sent in val_original_src_dataset]

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    translation_map = {}

    for i in tqdm(range(len(original_src_dataset))):
        key = tokenizer.decode(
            tokenizer(src_dataset[i], truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        translation_map[key.strip()] = {"src": original_src_dataset[i], "tgt": tgt_dataset[i]}

    for i in tqdm(range(len(val_original_src_dataset))):
        key = tokenizer.decode(
            tokenizer(val_src_dataset[i], truncation=True, max_length=max_length, add_special_tokens=False)[
                "input_ids"
            ],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        translation_map[key.strip()] = {"src": val_original_src_dataset[i], "tgt": val_tgt_dataset[i]}

    trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=src_dataset,
        eval_prompts=val_src_dataset,
        config=config,
    )

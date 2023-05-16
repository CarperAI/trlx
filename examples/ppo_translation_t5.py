"""Example of using PPO to train a T5 model for translation.
Based on examples/summarize_daily_cnn/t5_summarize_daily_cnn.py"""

import json
import os
import sys
from typing import List

import torch
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
    import comet
    import evaluate

    if comet.__version__ != "1.1.3":
        raise ImportError
except ImportError:
    raise ImportError(
        "To run this example, please install `evaluate`, `nltk` and `comet==1.1.3` packages by "
        "running `pip install evaluate unbabel-comet==1.1.3`"
    )


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=612,
        epochs=100,
        total_steps=100000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=200,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        tracker="wandb",
    ),
    model=ModelConfig(
        model_path="t5-large",
        model_arch_type="seq2seq",
        num_layers_unfrozen=-1,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="t5-large",
        padding_side="right",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 2.0e-6,
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
        num_rollouts=256,
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
            "do_sample": False,
            "num_beams": 4,
            "temperature": 1.0,
        },
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # COMET is the metric we are optimizng for
    comet_metric = evaluate.load("comet", "wmt20-comet-da", progress_bar=False)
    bleu_metric = evaluate.load("bleu")
    chrf_metric = evaluate.load("chrf")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        original_sents = [translation_map[prompt.strip()] for prompt in prompts]

        scores = comet_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
            sources=[original["src"] for original in original_sents],
        )["scores"]

        # TODO: This is needed since there seems to be a bug in the comet metric
        # that changes torch's determinism setting. Remove this once the bug is fixed.
        torch.use_deterministic_algorithms(False, warn_only=True)
        return scores

    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        """Compute COMET, BLEU and CHRF for evaluation"""
        original_sents = [translation_map[prompt.strip()] for prompt in prompts]

        comet_score = comet_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
            sources=[original["src"] for original in original_sents],
        )["mean_score"]

        bleu_score = bleu_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
        )["bleu"]

        chrf_score = chrf_metric.compute(
            predictions=[output.strip() for output in outputs],
            references=[original["tgt"] for original in original_sents],
        )["score"]

        # TODO: This is needed since there seems to be a bug in the comet metric
        # that changes torch's determinism setting. Remove this once the bug is fixed.
        # Same issue as in `reward_fn`
        torch.use_deterministic_algorithms(False, warn_only=True)

        # For corpus-level metrics, it's better to ignore the sentence-level scores
        return {"bleu": bleu_score, "chrf": chrf_score, "comet": comet_score}

    # The WMT16 is large so we can benefit with using it as a streaming dataset
    train_dataset = load_dataset("wmt16", "de-en", split="train", streaming=True)
    valid_dataset = load_dataset("wmt16", "de-en", split="validation", streaming=True)

    src_lang = "en"
    tgt_lang = "de"
    PREFIX = "translate English to German: "

    # take 20,000 samples from the training set as prompts for training
    original_src_dataset = [sent_pair["translation"][src_lang] for sent_pair in train_dataset.take(20000)]
    tgt_dataset = [sent_pair["translation"][tgt_lang] for sent_pair in train_dataset.take(20000)]
    src_dataset = [PREFIX + src_sent for src_sent in original_src_dataset]

    # take 1,000 samples from the validation set as prompts for evaluation
    val_original_src_dataset = [sent_pair["translation"][src_lang] for sent_pair in valid_dataset.take(1000)]
    val_tgt_dataset = [sent_pair["translation"][tgt_lang] for sent_pair in valid_dataset.take(1000)]
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


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

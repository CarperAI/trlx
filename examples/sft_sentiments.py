import os
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline, GPTNeoXConfig

import trlx
from trlx.default_configs import default_sft_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main():
    config = default_sft_config()
    _20b_cfg = GPTNeoXConfig()
    _20b_cfg.num_hidden_layers = 80
    _20b_cfg.hidden_size = 8192
    _20b_cfg.num_attention_heads = 128
    _20b_cfg.intermediate_size = 32768

    config.train.seq_length = 512
    config.train.batch_size = 1
    config.train.total_steps = 200
    config.model.model_path = _20b_cfg
    # config.model.model_path = 'EleutherAI/gpt-neox-20b'
    config.tokenizer.tokenizer_path = 'EleutherAI/gpt-neox-20b'

    imdb = load_dataset("imdb", split="train+test")
    # Finetune on only positive reviews
    imdb = imdb.filter(lambda sample: sample["label"] == 1)

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    trainer = trlx.train(
        samples=imdb["text"],
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        metric_fn=metric_fn,
        config=config,
    )
    trainer.save_pretrained("reviews-sft")


if __name__ == "__main__":
    main()

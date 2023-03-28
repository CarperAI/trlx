import json
import os
import sys
from typing import Dict, List

from datasets import load_dataset
from datasets import Dataset
from transformers import pipeline
import pandas as pd

import trlx
from trlx.data.default_configs import TRLConfig, default_ilql_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ilql_config().to_dict(), hparams)
    config.model.model_path = 'nakcnx/OTG-Math-680'
    config.tokenizer.tokenizer_path = 'nakcnx/OTG-Math-680' #'nakcnx/TGPT-2-345M' #'nakcnx/TGPT-Neo-125M'

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "poom-sci/WangchanBERTa-finetuned-sentiment",
        top_k=3,
        truncation=True,
        batch_size=128, #256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    imdb = load_dataset("nakcnx/Thai-IMDB")
    df = pd.DataFrame( imdb['train'] )
    df = df.dropna()
    imdb2 = Dataset.from_pandas(df)
    
    trlx.train(
        samples=imdb2["review_th_pythainlp"],
        rewards=imdb2["sentiment"],
        eval_prompts=["หนังเรื่องไม่สนุกเลย"] * 32, #64,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

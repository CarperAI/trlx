import trlx

from datasets import load_dataset
from transformers import pipeline

if __name__ == "__main__":
    sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb")

    def metric_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True)
        sentiments = [output[1]["score"] for output in outputs]
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        samples=imdb["text"],
        rewards=imdb["label"],
        eval_prompts=["I don't know much about Hungarian underground"] * 64 + ["<|endoftext|>"] * 64,
        metric_fn=metric_fn
    )


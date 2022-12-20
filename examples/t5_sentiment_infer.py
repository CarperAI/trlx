import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from datetime import datetime, timedelta
import pandas as pd
from accelerate import Accelerator
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()
from datasets import load_dataset

def load_model_tokenizer(pretrained_path):
    led = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    return led, tokenizer

MAX_OUTPUT_LENGTH = 64
MAX_INPUT_LENGTH = 64


def inference(model, tokenizer, problem):
    gen_kwargs = {
       "max_length": MAX_OUTPUT_LENGTH,
       "do_sample": True,
       "num_return_sequences": 1
    }
    model.eval()
    model.to(device)

    batch = tokenizer(
         problem,
         padding="max_length",
         truncation=True,
         max_length=MAX_INPUT_LENGTH,
    )


    batch['input_ids'] = [batch['input_ids'], batch['input_ids']]
    batch['attention_mask'] = [batch['attention_mask'], batch['attention_mask']]
    batch['input_ids'] = torch.tensor(batch['input_ids']).to(device)
    batch['attention_mask'] = torch.tensor(batch['attention_mask']).to(device)


    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )

        generated_tokens = accelerator.pad_across_processes(
              generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
    solutions = tokenizer.batch_decode(generated_tokens)
    return solutions


def main():
    accelerator = Accelerator()
    pretrained = "flan-t5-sentiment/checkpoint-5000"
    model, tokenizer = load_model_tokenizer(pretrained)
    model.to(device)
    
    imdb = load_dataset("imdb", split="test").select(range(100))

    start = time.time()
    prompts = ["Generate review for IMDB film start with: " + \
                        " ".join(review.split()[:4]) for review in imdb["text"]]
    for i in range(10):
        sols = inference(model, tokenizer, prompts[i])
        
        for sol in sols:
            sol = sol.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
            print("Prompt: ", prompts[i].strip())
            print("Review: ", sol.strip())
        print("=======================================")


if __name__ == "__main__":
    main()
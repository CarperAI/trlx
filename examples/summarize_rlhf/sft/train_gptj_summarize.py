import random
import numpy as np
import torch
from datasets import load_metric
import evaluate
import os
import wandb
wandb.init(project="gpt2-supervised-summarize", entity="pvduy")
from examples.summarize_rlhf.sft.summarize_dataset import TLDRDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
# deepspeed train_gptneo_summarize.py --config ds_config_gpt_neo_27.json
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator
)


def main():
    output_dir = "gptj-supervised-summarize-checkpoint"
    train_batch_size = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 800
    max_input_length = 550
    save_steps = 1600
    num_train_epochs = 5
    random.seed(42)
    # Load the GPT tokenizer.
    from transformers import AutoTokenizer, AutoModelForCausalLM
 
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    train_dataset = TLDRDataset("pvduy/openai_summarize_tldr", tokenizer, "train_sft", max_length=max_input_length)
    dev_dataset = TLDRDataset("pvduy/openai_summarize_tldr", tokenizer, "valid" ,max_length=max_input_length)
    rouge = evaluate.load('rouge')
    
    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    training_args = TrainingArguments(
        output_dir=output_dir, 
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        deepspeed='./ds_config_gptj.json'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    trainer.train()
    trainer.save_model(output_dir)

if __name__=="__main__":
    main()

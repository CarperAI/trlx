from torch.utils.data import dataset
from datasets import load_dataset

from datasets import load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
import os
from summarize_dataset import get_dataset_from_jsonl

val_openai_summ, val_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "valid.jsonl"), False)

MAX_LENGTH_INPUT = 512
MAX_LENGTH_OUTPUT = 64

class TLDRDataset(dataset.Dataset):

    def __init__(self, tokenizer, type_data='train'):
        if type_data == 'train':
            self.prompts, self.outputs = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "train.jsonl"), False)
        else:
            self.prompts, self.outputs = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/sandbox/trlx/openai_data/tldr_filtered", "valid.jsonl"), False)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        input_text = "Please summarize this document: " + self.prompts[idx]
        output_text = self.outputs[idx]

        model_input = self.tokenizer(
            input_text,
            max_length=MAX_LENGTH_INPUT, 
            padding='max_length',
            truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                output_text,
                max_length=MAX_LENGTH_OUTPUT,
                padding='max_length',
                truncation=True
            )["input_ids"]
            model_input['labels'] = labels
            model_input['labels'] = [-100 if token == self.tokenizer.pad_token_id else token for token in model_input['labels']]
        return model_input
    
import wandb 
wandb.init(name="flan-t5-summarize", project="add_t5", entity="pvduy")



if __name__=="__main__":
    config = {
        "logging_steps": 10,
        "eval_steps": 2000,
        "save_steps": 4000,
        "batch_size": 4,
        "batch_size_val": 4,
        "warmup_steps": 100,
        "accum_steps": 4,
        "num_beams": 3,
        "output_dir": "flan-t5-summarize",
    }

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str,
        )

        return rouge_output

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        do_train=True,
        num_train_epochs=5,
        do_eval=True,
        predict_with_generate=True,
        evaluation_strategy="steps",
        adam_beta1=0.9,
        adam_beta2=0.999,
        learning_rate=5e-5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size_val"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        eval_accumulation_steps=1,
        fp16=True,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=config["accum_steps"],
        deepspeed='ds_config_trlx_t5.json',
    )
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    rouge = load_metric("rouge")
    
    train_dataset = TLDRDataset(tokenizer, type_data='train')
    val_dataset = TLDRDataset(tokenizer, type_data='test')

    train_dataset[0]

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {params}")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

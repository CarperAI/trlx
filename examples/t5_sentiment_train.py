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

MAX_LENGTH_INPUT = 16
MAX_LENGTH_OUTPUT = 48

class IMDBDataset(dataset.Dataset):

    def __init__(self, tokenizer, type_data='train'):
        if type_data == 'train':
            imdb = load_dataset("imdb", split="train")
        else:
            imdb = load_dataset("imdb", split="test").select(range(100))
        print(len(imdb))
        #self.prompts = ["Generate review for IMDB film start with: " + \
        self.prompts = ["Generate a review film start with: " + " ".join(review.split()[:4]) for review in imdb["text"]]
        self.outputs = [" ".join(review.split()[4:]).replace("<br />", "") for review in imdb["text"]]
        for i in range(10):
            print('input: ', self.prompts[i])
            print('output: ', self.outputs[i])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        input_text = self.prompts[idx]
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
wandb.init(name="flan-t5-sentiment", project="add_t5", entity="pvduy")



if __name__=="__main__":
    config = {
        "logging_steps": 10,
        "eval_steps": 1000,
        "save_steps": 2000,
        "batch_size": 32, 
        "batch_size_val": 32,
        "warmup_steps": 100,
        "accum_steps": 1,
        "num_beams": 5,
        "output_dir": "flan-t5-sentiment-train-only",
    }

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        # bleu_output = bleu.compute(predictions=pred_str, references=label_str).mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4)
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        do_train=True,
        num_train_epochs=3,
        do_eval=True,
        predict_with_generate=True,
        evaluation_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size_val"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        save_total_limit=2,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=config["accum_steps"],
        # deepspeed='ds_config_trlx_t5.json',
    )
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    #model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-review/checkpoint-3000/")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    rouge = load_metric("rouge")
    
    train_dataset = IMDBDataset(tokenizer, type_data='train')
    val_dataset = IMDBDataset(tokenizer, type_data='test')

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

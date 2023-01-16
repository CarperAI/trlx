from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

import wandb
from tic_tac_toe_data import generate_dataset


def main() -> None:
    
    # Create the train dataset
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B",)# mask_token="<mask>") #Why is this here?
    list_of_game_strings = generate_dataset(10)
    train_dataset = Dataset.from_dict({"text":list_of_game_strings})
    tokenized_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    # create labels by copying input_ids column to labels column
    lm_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    print(lm_dataset)
    print(tokenizer.decode(lm_dataset[1]["input_ids"]))


    wandb.login()

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    training_args = TrainingArguments(per_device_train_batch_size=1,output_dir=".checkpoints", evaluation_strategy="epoch")
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()

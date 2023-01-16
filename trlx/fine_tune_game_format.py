from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

import wandb
from trlx.tic_tac_toe_data import generate_dataset


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    
    # Create the train dataset
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", mask_token="<mask>")
    list_of_game_strings = generate_dataset(10)
    train_dataset = Dataset.from_dict({"text":list_of_game_strings})
    tokenized_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    # create labels by copying input_ids column to labels column
    lm_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    print(lm_dataset)
    print(tokenizer.decode(lm_dataset[1]["input_ids"]))


    wandb.login()


    training_args = TrainingArguments(output_dir=".checkpoints", evaluation_strategy="epoch")
    
    # Convert the strings to features that can be fed into a model
    
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

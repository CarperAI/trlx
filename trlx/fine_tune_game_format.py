from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import wandb
from trlx.tic_tac_toe_data import generate_dataset


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    
    # Create the dataset
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", mask_token="<mask>")
    datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    dataset = generate_dataset(10)
    print(dataset["train"][0])
    
    # wandb.login()


    # training_args = TrainingArguments(output_dir=".checkpoints", evaluation_strategy="epoch")
    
    # # Convert the strings to features that can be fed into a model

    # # Convert the features to a torch dataset
    # dataset = torch.utils.data.TensorDataset(*(f.input_ids for f in features))
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     # eval_dataset=small_eval_dataset,
    #     # compute_metrics=compute_metrics,
    # )
    
    # trainer.train()

if __name__ == "__main__":
    main()

import wandb
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from soft_optim.game_generator import generate_games


def create_dataset(tokenizer: AutoTokenizer, number_games: int = 10) -> Dataset:
    """Create the dataset
    
    This is a collection of full game prompts (tokenized).

    Args:
        tokenizer: Tokenizer
        number_games: Number of games

    Returns:
        Dataset: Full game prompts dataset
    """
    # Create the dataset from a list of game strings
    list_of_game_strings = generate_games(number_games)
    dataset = Dataset.from_dict({"text":list_of_game_strings})
    
    # Tokenize the text prompts (creates "input_ids" property for each dataset item)
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    
    # Set the labels to be the same as the input IDs
    dataset = dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
    
    return dataset


def main(model_name: str = "gpt2") -> None:
    """Fine tune a language model on the games dataset
    
    This is so that our model reliably outputs allowed game moves.
    """
    # Create the tokenized dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = create_dataset(tokenizer)
   
    # Initialise Weights & Biases
    wandb.login()

    # Create the model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    training_args = TrainingArguments(output_dir=".checkpoints", evaluation_strategy="epoch")
    
    # Fine tune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save the final state dictionary

if __name__ == "__main__":
    main()

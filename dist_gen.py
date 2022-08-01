import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from trl.gpt2 import GPT2HeadWithValueModel
from torch.optim import Adam

def main():
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
    optimizer = Adam(model.parameters(), lr=1.41e-5)
    model.config.pad_token_id = model.config.eos_token_id
    model = accelerator.prepare(model)
    model.to(accelerator.device)
    print(accelerator.state)
    accelerator.print(model)

    optimizer = accelerator.prepare(optimizer)
    
    rank = torch.distributed.get_rank()
    if rank == 0:
        text_in = "The purpose of life is "
    elif rank == 1:
        text_in = "Are you human? "

    batch = tokenizer(text_in, return_tensors="pt").to(accelerator.device)
    
    # had to run this 1 time at the start else was giving device mismatch error.
    # So, before directly using `model.generate` pass a batch with dummy data through the model 
    outputs = model(**batch)
    
    print(batch)
    gen_kwargs = {
        "max_length": 64,
        "num_beams": 10,
        "min_length": 20,
        "length_penalty": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
    }
    with torch.no_grad():
        unwrapped_model = accelerator.unwrap_model(model)
        # synced_gpus was necessary else resulted into indefinite hang
        outputs = unwrapped_model.generate(batch["input_ids"], synced_gpus=True, **gen_kwargs)

    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nrank{rank}:\n   in={text_in}\n  out={text_out}")

    # Now backprop loss


    
if __name__ == "__main__":
    main()

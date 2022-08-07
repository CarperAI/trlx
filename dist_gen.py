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
from transformers import DataCollatorForLanguageModeling
import torch.nn.functional as F


def main():
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    ref_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    optimizer = Adam(model.parameters(), lr=1.41e-5)
    model.config.pad_token_id = model.config.eos_token_id
    model = accelerator.prepare(model)
    model.to(accelerator.device)
    print(accelerator.state)


    optimizer = accelerator.prepare(optimizer)

    rank = torch.distributed.get_rank()
    if rank == 0:
        text_in = "The purpose of life is "
    elif rank == 1:
        text_in = "Are you human? "

    query_tensors = tokenizer(text_in, return_tensors="pt").to(accelerator.device)["input_ids"]

    # had to run this 1 time at the start else was giving device mismatch error.
    # So, before directly using `model.generate` pass a batch with dummy data through the model
    outputs = model(query_tensors)

    print(query_tensors)
    gen_kwargs = {
        "max_length": 64,
        "min_length": 20,
    }
    with torch.no_grad():
        unwrapped_model = accelerator.unwrap_model(model)
        # synced_gpus was necessary else resulted into indefinite hang
        response_tensors = unwrapped_model.generate(query_tensors, synced_gpus=True, **gen_kwargs)

    text_out = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
    # Arbitrarily score generation
    score = torch.tensor([1.0]).to(accelerator.device)
    print(f"\nrank{rank}:\n   in={text_in}\n  out={text_out}")

    # Now compute ppo loss
    ## First compute logprobs and ref_logprobs
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    input_ids = collator([torch.cat([q, r]) for q, r in zip(query_tensors, response_tensors)])["input_ids"]

    with torch.no_grad():
        logits, _, v = model(input_ids)
        #print('values', v)
        ref_logits, _, _ = ref_model(input_ids.cpu())
        ref_logits = ref_logits.to(accelerator.device)

    logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
    ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])

    # Only care about logprobs for generated text
    start = query_tensors.size()[-1] - 1
    end = query_tensors.size()[-1] + response_tensors.size()[-1] - 1
    logprobs = logprobs[:, start:end]
    ref_logprobs = ref_logprobs[:, start:end]
    v = v[:, start-1: end-1]
    print('logprob sizes', logprobs.size(), ref_logprobs.size(), v.size())


    ## Compute rewards
    kl = logprobs - ref_logprobs
    non_score_reward = .2 * kl
    reward = non_score_reward.clone()
    reward[-1] += score

    ## Compute losses
    lastgaelam = 0
    advantages_reversed = []
    gen_len = response_tensors.shape[1]
    for t in reversed(range(gen_len)):
        nextvalues = v[:, t+1] if t < gen_len - 1 else 0.0
        delta = reward[:, t] + 1.00 * nextvalues - v[:, t]
        lastgaelam = delta + 1.00 * .99 * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

    returns = advantages + v
    advantages = advantages.detach()

    ### With grad this time
    logits, _, vpred = model(input_ids)
    logprob = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    logprob, vpred = logprob[:, -gen_len:], vpred[:, -gen_len-1:-1]
    vf_loss = torch.mean((vpred - returns)**2)

    # Backpropagate
    optimizer.zero_grad()
    accelerator.backward(vf_loss)
    optimizer.step()


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


if __name__ == "__main__":
    main()

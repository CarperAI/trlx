import argparse
import os
from string import Template

import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--base_model", type=str, required=True, help="Path to HF checkpoint with the base model")

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to either a local directory or a HF checkpoint with reward model's weights",
)

parser.add_argument("--revision", type=str, required=False, help="Optional branch/commit of the HF checkpoint")

parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

model_name = args.checkpoint.split("/")[-1]
device = torch.device(args.device)


class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns


if os.path.isdir(args.checkpoint):
    directory = args.checkpoint
else:
    directory = snapshot_download(args.checkpoint, revision=args.revision)

print(f"searching through {os.listdir(directory)} in {directory}")

for fpath in os.listdir(directory):
    if fpath.endswith(".pt") or fpath.endswith(".bin"):
        checkpoint = os.path.join(directory, fpath)
        break

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = RewardModel(args.base_model, tokenizer.eos_token_id)
model.load_state_dict(torch.load(checkpoint))
model.eval()
model.requires_grad_(False)
model = model.half().to(device)

input = tokenizer("reward model's hash", return_tensors="pt").to(device)
print(f"{model(input.input_ids)=}")

traced_script_module = torch.jit.trace(model, input.input_ids)

os.makedirs(f"model_store/{model_name}/1", exist_ok=True)
traced_script_module.save(f"model_store/{model_name}/1/traced-model.pt")

config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "triton_config.pbtxt")
with open(config_path) as f:
    template = Template(f.read())
config = template.substitute({"model_name": model_name})
with open(f"model_store/{model_name}/config.pbtxt", "w") as f:
    f.write(config)

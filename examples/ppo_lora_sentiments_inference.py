"""
Example script for inferencing a CausalLM model trained in `trlx` with LoRA modules.

Usage:

python examples/ppo_lora_sentiments_inference.py --checkpoint_dir /path/to/checkpoint

NOTE: Your training script must call `trainer.save_pretrained` to save the LoRA weights
to the checkpoint directory.
"""
import argparse
import os

import torch
from opendelta import AutoDeltaConfig, AutoDeltaModel
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True)
args = parser.parse_args()

config = TRLConfig.load_yaml(os.path.join(args.checkpoint_dir, "trainer_config.yaml"))

# `model_path` is the path passed to `trainer.save_pretrained()` during training. Default: "ckpts/hf"
model_path = os.path.join(args.checkpoint_dir, "delta")
model = AutoModelForCausalLMWithHydraValueHead.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the delta configs, inject LoRA modules into the model, and load the actual LoRA weights
delta_config = AutoDeltaConfig.from_dict(config.model.delta_kwargs)
delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=model)
delta_weights = torch.load(str(os.path.join(model_path, "delta_pytorch_model.bin")))
# TODO: The delta keys should only contain the LoRA modules and value heads
print(delta_weights.keys())
# Print the delta weights size in MB ignoring value head keys
print(f"Delta weights size: {sum([v.numel() * v.element_size() for k, v in delta_weights.items() if 'v_head' not in k]) / 1e6} MB")
model.base_model.load_state_dict(delta_weights, strict=False)

prompts = [
    "That film was an",
    "I'd rate the movie",
    "I wasn't familiar with the director's previous work but"
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
tokens = model.generate(**inputs, **config.method.gen_kwargs)
text = tokenizer.batch_decode(tokens, skip_special_tokens=True)

# "Pretty" print the results
print("=" * 60)
print("\nGenerated Text:\n")
for t in text:
    print(f"⇥ {t}\n")
print("=" * 60)

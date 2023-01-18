# flake8: noqa
import pdb

import torch
import transformers

from trlx.trainer.nn.ilql_models import AutoModelForCausalLMWithILQLHeads
from trlx.trainer.nn.ppo_models import (
    AutoModelForCausalLMHydraWithValueHead,
    AutoModelForCausalLMWithValueHead,
)

save_dir = "ptm-test"

################################################################################
# Memory Tracking
################################################################################

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


################################################################################
# PPO (No Hydra)
################################################################################

# Load pretrained model off the Hub
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model.v_head[2].bias = torch.nn.Parameter(torch.tensor([69.0]))
model.save_pretrained("test-save-gpt2-ppo")

loaded_model = AutoModelForCausalLMWithValueHead.from_pretrained("test-save-gpt2")

# Check if the loaded model state dict is the same as the original model state dict
# This is a sanity check to make sure the model is saved and loaded correctly

original = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
print(model.v_head[2].bias)
print(loaded_model.v_head[2].bias)
print(original.v_head[2].bias)
for k, v in model.state_dict().items():
    assert torch.all(v == loaded_model.state_dict()[k]), print(
        "❌ State Dicts Are Not Equal:", (k, v, loaded_model.state_dict()[k])
    )

print("✅ State Dicts Are Equal")

model = AutoModelForCausalLMHydraWithValueHead.from_pretrained("gpt2")
model.v_head[2].bias = torch.nn.Parameter(torch.tensor([69.0]))
# model.save_pretrained("test-save-gpt2-ppo")
model.push_to_hub("jon-tow/ppo-test")


################################################################################
# PPO (Hydra)
################################################################################

# loaded_model = AutoModelForCausalLMHydraWithValueHead.from_pretrained("jon-tow/ppo-test")

# # Check if the loaded model state dict is the same as the original model state dict
# # This is a sanity check to make sure the model is saved and loaded correctly

# original = AutoModelForCausalLMHydraWithValueHead.from_pretrained("gpt2")
# print(model.v_head[2].bias)
# print(loaded_model.v_head[2].bias)
# print(original.v_head[2].bias)
# for k, v in model.state_dict().items():
#     assert torch.all(v == loaded_model.state_dict()[k]), \
#         print("❌ State Dicts Are Not Equal:", (k, v, loaded_model.state_dict()[k]))

# print('✅ State Dicts Are Equal')


################################################################################
# ILQL
################################################################################

model = AutoModelForCausalLMWithILQLHeads.from_pretrained(pretrained_model)
print(f"{'=' * 40}")
print_gpu_utilization()
print(f"{'=' * 40}")

print(model.ilql_heads.q_heads[0][0].bias)
model.ilql_heads.q_heads[0][0].bias = torch.nn.Parameter(
    torch.ones_like(model.ilql_heads.q_heads[0][0].bias) * 49.9
)
model.save_pretrained(save_dir)

loaded_model = AutoModelForCausalLMWithILQLHeads.from_pretrained(save_dir)

# Check if the loaded model state dict is the same as the original model state dict
# This is a sanity check to make sure the model is saved and loaded correctly

original = AutoModelForCausalLMWithILQLHeads.from_pretrained("gpt2")
print("model q heads", model.ilql_heads.q_heads[0][0].bias)
print("loaded model q heads", loaded_model.ilql_heads.q_heads[0][0].bias)
print("fresh model q heads", original.ilql_heads.q_heads[0][0].bias)
for k, v in model.state_dict().items():
    assert torch.all(v == loaded_model.state_dict()[k]), print(
        "❌ State Dicts Are Not Equal:", (k, v, loaded_model.state_dict()[k])
    )

print("✅ State Dicts Are Equal")

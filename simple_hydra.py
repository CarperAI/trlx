import torch
from torch import nn
from copy import deepcopy

######Defining Model######

class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.a1 = nn.GELU()
    
    def forward(self, x):
        return self.a1(self.linear(x))

class ModelBranch(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.block = deepcopy(trunk.block2)
        self.trunk = trunk

    def forward(self, x):
        _, mid_output = self.trunk(x, return_all=True)
        return self.block(mid_output)

class ModelTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(10, 10)
        self.block2 = Block(10, 10)

    def forward(self, x, return_all=False):
        out = self.block1(x)
        output = (self.block2(out), out) if return_all else self.block2(out)
        return output

class SimpleHydra(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = ModelTrunk()
        self.branch = ModelBranch(self.trunk)

    def forward(self, x):
        return self.trunk(x)


######Training Model######

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import broadcast
from torch.utils.data import DataLoader
import torch
from trlx.model.nn.ppo_models import GPTHydraHeadWithValueModel
from trlx.data.configs import TRLConfig

accelerator = Accelerator()

config = TRLConfig.load_yaml("configs/ppo_config.yml")
#model = AutoModelForCausalLM.from_pretrained('gpt2')
unwrapped_model = SimpleHydra()

input = torch.rand(256, 10)


def collate_fn(elems):
	return torch.stack(elems)

dataloader = DataLoader(input, 32, collate_fn=collate_fn)

model, dataloader = accelerator.prepare(unwrapped_model, dataloader)

data = iter(dataloader)
batch = next(data)
print("batch device", batch.get_device())  # Tensor has get_device function
# Module does not have get_device()
# Why does trunk linear have two separate devices despite zero 3...
print("trunk linear weight", unwrapped_model.trunk.block1.linear.weight.get_device())
output = model(batch)
ref_output = unwrapped_model.branch.forward(batch)
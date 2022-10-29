import torch
from torch import nn

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
        self.block = Block(10, 10)
        self.trunk = trunk

    def forward(self, x):
        _, mid_output = self.trunk(x)
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

block = Block(10, 10)

ModelBranch(block)

SimpleHydra()

layer_norm = torch.nn.LayerNorm(10, 1e-5)

print(layer_norm)
print(layer_norm.weight, layer_norm.bias)
print(layer_norm.parameters())
for param in layer_norm.parameters():
    print(param)
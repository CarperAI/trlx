import torch
from torch import nn
from copy import deepcopy

from trlx.model.nn.ppo_models import ModelBranch
from trlx.data.configs import TRLConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.utils import broadcast
from torch.utils.data import DataLoader
import torch
from trlx.model.nn.ppo_models import GPTHydraHeadWithValueModel, PretrainedHydraModel



config = TRLConfig.load_yaml("configs/ppo_config.yml")

######Defining Model######

class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.a1 = nn.GELU()

    def forward(self, x):
        return self.a1(self.linear(x))

class SimpleModelBranch(nn.Module):
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




def simple_hydra():

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


def z3_model_branch():
    # This sucessfully does a forward pass on both trunk and ref model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    input = ['Hello world']*256


    def collate_fn(elems):
        input = tokenizer(elems, return_tensors="pt")
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    dataloader = DataLoader(input, 2, collate_fn=collate_fn)
    #trunk = AutoModelForCausalLM.from_pretrained('gpt2')
    #hf_config = trunk.config
    #branch = ModelBranch(hf_config, trunk, 2)
    branch = PretrainedHydraModel('lvwerra/gpt2-imdb', 2)
    gpt_blocks = branch.gpt.transformer.h
    gpt_blocks_to_freeze = list(gpt_blocks)[:-2]
    for m in gpt_blocks_to_freeze:
        m.requires_grad_(False)

    opt = torch.optim.AdamW(
        branch.parameters(),
        lr=4.5e-4,
        betas=[0.9, 0.95],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        10000,
        4.5e-4,
    )

    accelerator = Accelerator()
    branch, dataloader, opt, scheduler = accelerator.prepare(branch, dataloader, opt, scheduler)

    '''# Generation hangs when inputs on different ranks are not the same
    text = "hello world" if torch.distributed.get_rank() == 0 else "goodbye my sweet prince"
    tokenizer.pad_token = tokenizer.eos_token
    dummy_input = tokenizer([text], padding="max_length", truncation=True, max_length=12, return_tensors='pt')['input_ids'].to(accelerator.device)
    gen_kwargs = {'max_length': 24, 'min_length': 24}
    print(dummy_input.size())
    out = branch.generate(dummy_input, **gen_kwargs)
    print(out.size())
    decoded_out = tokenizer.batch_decode(out)
    print(decoded_out)
    exit()'''

    # Seem to be able to backprop with no loop
    for i, batch in enumerate(iter(dataloader)):
        print("STEP", i)

        # Issue seems to be with calling generate or no grad forward around backprop
        #with torch.no_grad():
        #out = branch.generate(**batch)  # Also cannot call .generate around forward/
        #decoded_out = tokenizer.batch_decode(out)
        with torch.no_grad():
            branch.eval()
            out = branch.generate(**batch)  # Also cannot call .generate around forward/
            decoded_out = tokenizer.batch_decode(out)
            _ = branch(**batch)  #  Issue with calling no grad forward around backward

        branch.train()
        logits, vpred, ref_logits = branch(**batch)
        loss = logits.sum()
        accelerator.backward(loss)
        accelerator.wait_for_everyone()

    print("EXITING")


if __name__ == "__main__":
    #simple_hydra()
    z3_model_branch()
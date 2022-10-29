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
unwrapped_model = GPTHydraHeadWithValueModel(
            config.model.model_path, config.model.num_layers_unfrozen
        )
tokenizer = AutoTokenizer.from_pretrained('gpt2')

input = ['Hello world']*256


def collate_fn(elems):
	input = tokenizer(elems, return_tensors="pt")
	input_ids = input['input_ids']
	attention_mask = input['attention_mask']
	return input_ids, attention_mask

dataloader = DataLoader(input, 128, collate_fn=collate_fn)

model, dataloader = accelerator.prepare(unwrapped_model, dataloader)

#print(unwrapped_model.gpt.h)

#unwrapped_model.ref_model.parallelize()
print(len(dataloader))

#if accelerator.is_main_process:
data = iter(dataloader)
input_ids, attention_mask = next(data)
gen_kwargs = {'max_length': 24, 'min_length': 24, 'top_k': 0.0, 'top_p': 1.0, 'do_sample': True}
input_ids = input_ids.to(accelerator.device)
attention_mask = attention_mask.to(accelerator.device)
print("GENERATING")
# Generating seems to freeze if only one process is doing it
# Generation also runs when using hydra model
# Generation runs  when using gpt2!
# Printing and barriers seem to sometimes cause deadlocks
out = model.generate(input_ids, **gen_kwargs)
print("FINISHED GENERATING")
decoded_out = tokenizer.batch_decode(out)
print(len(decoded_out))

with torch.no_grad():
	print("COMPUTING LOGITS")
	logits, _, v, ref_logits = model(input_ids)
	#print("COMPUTING REF LOGITS")
	#ref_logits = unwrapped_model.ref_model(
	#					input_ids, return_dict=False
	#				)

# Testing broadcasting
'''
new_data = [None]  # Note lists must be equal sizes
if accelerator.is_main_process:
	new_data = [torch.tensor(1)]
	#torch.distributed.broadcast(new_data)  This does not work becuase must be tensor
#device = torch.device("cpu")
cur_device = torch.cuda.current_device()
print(cur_device)
torch.cuda.set_device(cur_device)
torch.distributed.broadcast_object_list(new_data, src=0)  # Device type must be cuda for nccl backend

print(new_data)
print(new_data[0].device)
'''
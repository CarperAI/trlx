from transformers import GPTJForCausalLM
from transformers import AutoTokenizer
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B"
)
model.to(device)

prompt = (
'''Play a game of tic-tac-toe:
- - -
- x -
- - -

o - -
- x -
- - -

o - -
- x -
x - -

o - o
- x -
x - -

o - o
- x -
x x -

o o o
- x -
x x -

o wins!

Play a game of tic-tac-toe:'''
)

input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print("finished tokenizing, starting generating")
output_tokens = model.generate(input_tokens, do_sample=True, temperature=1,max_length=200)
print("finished generating")
text = tokenizer.batch_decode(output_tokens)[0]
print(text)
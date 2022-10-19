[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

# Transformer Reinforcement Learning X

`trlx` allows you to fine-tune 🤗 Huggingface supported language models (`gpt2`, `gpt-j`, `gpt-neo` and `gpt-neox` based) up to 20B parameters using reinforcement learning via either a provided reward function or reward-labeled dataset. Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1909.08593.pdf)) and Implicit Language Q-Learning ([ILQL](https://sea-snell.github.io/ILQL_site/)) are implemented.

## Train

```python
import trlx

# optimize some reward function
model = trlx.train('gpt2', reward_fn=lambda samples: [sample.count('cats') for sample in samples])

# or steer a model with a collection of rated samples
model = trlx.train('EleutherAI/gpt-j-6B', dataset=[('dogs', 'geese'), (1.0, 100.0)])

# model is a wrapper with some logit preprocessing
model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

Launch distributed training with 🤗 Accelerate

```bash
accelerate config
accelerate launch examples/ilql_simulacra.py
```

For more usage see [examples](./examples)

## Install
```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113 # for cuda
pip install -e .
```

For development also do:
```
pip install -e ".[dev]"
pre-commit install # see .pre-commit-config.yaml
```

and check our [docs](https://trlX.readthedocs.io)

## Acknowledgements

Thanks Leandro for building the original [trl](https://github.com/lvwerra/trl/)

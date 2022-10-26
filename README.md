[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

# Transformer Reinforcement Learning X

[![Docs Status][docs-image]][docs-url]

**[Documentation](https://trlX.readthedocs.io)**

`trlx` allows you to fine-tune 🤗 Hugging Face supported language models (`gpt2`, `gpt-j`, `gpt-neo` and `gpt-neox` based) up to 20B parameters using reinforcement learning via either a provided reward function or reward-labeled dataset. Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1909.08593.pdf)) and Implicit Language Q-Learning ([ILQL](https://sea-snell.github.io/ILQL_site/)) are implemented.

You can read more about trlX in our [documentation](https://trlX.readthedocs.io).

## Installation
### From Source
```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113 # for cuda
pip install -e .
```

## How to Train
You can train your model using a reward function or a reward-labeled dataset.

### Using a reward function
```python
import trlx

# optimize some reward function
model = trlx.train('gpt2', reward_fn=lambda samples: [sample.count('cats') for sample in samples])

# model is a wrapper with some logit preprocessing
model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

### Using a reward-labeled dataset

```python
import trlx

# Steer a model with a collection of rated samples
model = trlx.train('EleutherAI/gpt-j-6B', dataset=[('dolphins', 'geese'), (1.0, 100.0)])

# model is a wrapper with some logit preprocessing
model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

### Using 🤗 Accelerate to speed up the training
Launch distributed training with 🤗 Accelerate (only DeepSpeed integration is tested)

```bash
accelerate config
accelerate launch examples/simulacra.py
```

For more usage see [examples](./examples)

## Contributing

For development check out these [guidelines](./CONTRIBUTING.md)
and also read our [docs](https://trlX.readthedocs.io)

## Acknowledgements

Thanks Leandro for starting the original [trl](https://github.com/lvwerra/trl/)

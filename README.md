[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

# Transformer Reinforcement Learning X

trlX allows you to fine-tune 🤗 Hugging Face supported language models (`gpt2`, `gpt-j`, `gpt-neo` and `gpt-neox` based) up to 20B parameters using reinforcement learning via either a provided reward function or reward-labeled dataset. Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1909.08593.pdf)) and Implicit Language Q-Learning ([ILQL](https://sea-snell.github.io/ILQL_site/)) are implemented.

You can read more about trlX in our [documentation](https://trlX.readthedocs.io).

Want to collect human annotations for your RL application? Check out [CHEESE!](https://github.com/carperai/cheese), our library for HiTL data collection.

## Installation
```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
```

## How to Train
You can train a model using a reward function or a reward-labeled dataset.

#### Using a reward function
```python
trainer = trlx.train('gpt2', reward_fn=lambda samples: [sample.count('cats') for sample in samples])
```
#### Using a reward-labeled dataset
```python
trainer = trlx.train('EleutherAI/gpt-j-6B', dataset=[('dolphins', 'geese'), (1.0, 100.0)])
```

#### Trained model is a wrapper over a given autoregressive model
```python
trainer.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

#### Use 🤗 Accelerate to launch distributed training

```bash
accelerate config # choose DeepSpeed option
accelerate launch examples/simulacra.py
```

#### Use Ray Tune to launch hyperparameter sweep
```bash
python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml examples/ppo_sentiments.py
```

For more usage see [examples](./examples)

## Contributing

For development check out these [guidelines](./CONTRIBUTING.md)
and also read our [docs](https://trlX.readthedocs.io)

## Acknowledgements

Many thanks to Leandro von Werra for contributing with [trl](https://github.com/lvwerra/trl/), a library that initially inspired this repo.

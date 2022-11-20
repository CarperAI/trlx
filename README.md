[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

# Transformer Reinforcement Learning X

trlX allows you to fine-tune 🤗 Hugging Face supported language models (`gpt2`, `gpt-j`, `gpt-neo` and `gpt-neox` based) up to 20B parameters using reinforcement learning via either a provided reward function or reward-labeled dataset. Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1909.08593.pdf)) and Implicit Language Q-Learning ([ILQL](https://sea-snell.github.io/ILQL_site/)) are implemented.

You can read more about trlX in our [documentation](https://trlX.readthedocs.io).

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
model = trlx.train('gpt2', reward_fn=lambda samples: [sample.count('cats') for sample in samples])
```
#### Using a reward-labeled dataset
```python
model = trlx.train('EleutherAI/gpt-j-6B', dataset=[('dolphins', 'geese'), (1.0, 100.0)])
```

#### Trained model is a wrapper over a given autoregressive model
```python
model.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

#### Use 🤗 Accelerate to launch distributed training

```bash
accelerate config # choose DeepSpeed option
accelerate launch examples/simulacra.py
```

#### Use Ray Tune to launch hyperparameter sweep
```bash
python train_sweep.py --config configs/ray_tune_configs/ppo_config.yml --sweep-fn ppo_sentiments
```

For more usage see [examples](./examples)

## Contributing

For development check out these [guidelines](./CONTRIBUTING.md)
and also read our [docs](https://trlX.readthedocs.io)

## Acknowledgements

Many thanks to Leandro von Werra for hacking on the [trl](https://github.com/lvwerra/trl/)

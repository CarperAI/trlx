[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

[![DOI](https://zenodo.org/badge/545104023.svg)](https://zenodo.org/badge/latestdoi/545104023)

# Transformer Reinforcement Learning X

trlX allows you to fine-tune 🤗 Hugging Face supported language models of up to 20B parameters (such as `gpt2`, `gpt-j`, and `gpt-neox`, as well as T5 based models, including `google/t5-v1_1` and `google/flan-t5`)  using reinforcement learning via either a provided reward function or reward-labeled dataset. Proximal Policy Optimization ([PPO](https://arxiv.org/pdf/1909.08593.pdf)) and Implicit Language Q-Learning ([ILQL](https://sea-snell.github.io/ILQL_site/)) are implemented.

You can read more about trlX in our [documentation](https://trlX.readthedocs.io).

Want to collect human annotations for your RL application? Check out [CHEESE!](https://github.com/carperai/cheese), our library for HiTL data collection.

## Installation
```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
```

## Examples
For more usage see [examples](./examples). You can also try the colab notebooks below:
| Description      | Link |
| ----------- | ----------- |
| Simulacra Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vrmCLoHNlKvDVqJjMig-8tKDCfIEoym4?usp=sharing)|



## How to Train
You can train a model using a reward function or a reward-labeled dataset.

#### Using a reward function
```python
trainer = trlx.train('gpt2', reward_fn=lambda samples, **kwargs: [sample.count('cats') for sample in samples])
```
#### Using a reward-labeled dataset
```python
trainer = trlx.train('EleutherAI/gpt-j-6B', dataset=[('dolphins', 'geese'), (1.0, 100.0)])
```

#### Trainers provide a wrapper over their underlying model
```python
trainer.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

#### Save the resulting model to a Hugging Face pretrained language model. (Ready to upload to the Hub!)
```python
trainer.save_pretrained('/path/to/output/folder/')
```

🩹 Warning: Only the `AcceleratePPOTrainer` can write HuggingFace transformers to disk with `save_pretrained` at the moment, as ILQL trainers require inference behavior currently unsupported by available `transformers` architectures.

#### Use 🤗 Accelerate to launch distributed training

```bash
accelerate config # choose DeepSpeed option
accelerate launch examples/simulacra.py
```

#### Use Ray Tune to launch hyperparameter sweep
```bash
python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml examples/ppo_sentiments.py
```

## Logging

trlX uses the standard Python `logging` library to log training information to the console. The default logger is set to the `INFO` level, which means that `INFO`, `WARNING`, `ERROR`, and `CRITICAL` level messages will be printed to standard output.

To change the log level, you can use one of the direct setters. For example, to set the log level to `WARNING` you can use:

```python
import trlx

trlx.logging.set_verbosity_warning()
```

This will suppress `INFO` level messages, but still print `WARNING`, `ERROR`, and `CRITICAL` level messages.

You can also control logging verbosity by setting the `TRLX_VERBOSITY` environment variable to one of the standard logging [level names](https://docs.python.org/3/library/logging.html#logging-levels):

* `CRITICAL` (`trlx.logging.CRITICAL`)
* `ERROR` (`trlx.logging.ERROR`)
* `WARNING` (`trlx.logging.WARNING`)
* `INFO` (`trlx.logging.INFO`)
* `DEBUG` (`trlx.logging.DEBUG`)

```sh
export TRLX_VERBOSITY=WARNING
```

By default, [`tqdm`](https://tqdm.github.io/docs/tqdm/) progress bars are used to display training progress. You can disable them by calling `trlx.logging.disable_progress_bar()`, otherwise `trlx.logging.enable_progress_bar()` to enable.

Messages can be formatted with greater detail by setting `trlx.logging.enable_explicit_format()`. This will inject call-site information into each log which may be helpful for debugging.

```sh
[2023-01-01 05:00:00,000] [INFO] [ppo_orchestrator.py:63:make_experience] [RANK 0] Message...
```

> 💡 Tip: To reduce the amount of logging output, you might find it helpful to change log levels of third-party libraries used by trlX. For example, try adding `transformers.logging.set_verbosity_error()` to the top of your trlX scripts to silence verbose messages from the `transformers` library (see their [logging docs](https://huggingface.co/docs/transformers/main_classes/logging#logging) for more details).

## Contributing

For development check out these [guidelines](./CONTRIBUTING.md)
and also read our [docs](https://trlX.readthedocs.io)

## Acknowledgements

Many thanks to Leandro von Werra for contributing with [trl](https://github.com/lvwerra/trl/), a library that initially inspired this repo.

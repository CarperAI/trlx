[![EMNLP Paper](https://img.shields.io/badge/EMNLP_Paper-grey.svg?style=flat&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNjgiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDExKSIKICAgICBpZD0icmVjdDIxNzgiIC8+Cjwvc3ZnPgo=)](https://aclanthology.org/2023.emnlp-main.530/) [![DOI](https://zenodo.org/badge/545104023.svg)](https://zenodo.org/badge/latestdoi/545104023)  [![License](https://img.shields.io/github/license/CarperAI/trlx)](LICENSE)

# Transformer Reinforcement Learning X

trlX is a distributed training framework designed from the ground up to focus on fine-tuning large language models with reinforcement learning using either a provided reward function or a reward-labeled dataset.

Training support for 🤗 Hugging Face models is provided by [Accelerate](https://huggingface.co/docs/accelerate/)-backed trainers, allowing users to fine-tune causal and T5-based language models of up to 20B parameters, such as `facebook/opt-6.7b`, `EleutherAI/gpt-neox-20b`, and `google/flan-t5-xxl`. For models beyond 20B parameters, trlX provides [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)-backed trainers that leverage efficient parallelism techniques to scale effectively.

The following RL algorithms are currently implemented:

| Algorithm                                                                     | Accelerate Trainer | NeMo Trainer  |
|-------------------------------------------------------------------------------|:------------------:|:-------------:|
| [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1909.08593.pdf)    | ✅                 | ✅            |
| [Implicit Language Q-Learning (ILQL)](https://sea-snell.github.io/ILQL_site/) | ✅                 | ✅            |

📖 **[Documentation](https://trlX.readthedocs.io)**

🧀 **[CHEESE](https://github.com/carperai/cheese)** Collect human annotations for your RL application with our human-in-the-loop data collection library.

## Installation

```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

## Examples

For more usage see [examples](./examples). You can also try the colab notebooks below:
| Description | Link |
| ----------- | ----------- |
| Simulacra (GPT2, ILQL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarperAI/trlx/blob/main/examples/notebooks/trlx_simulacra.ipynb)|
| Sentiment (GPT2, ILQL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarperAI/trlx/blob/main/examples/notebooks/trlx_sentiments.ipynb)|

Latest runs of the examples are on our [Weights & Biases](https://wandb.ai/sorry/trlx-references/reportlist)

## How to Train

You can train a model using a reward function or a reward-labeled dataset.

#### Using a reward function

```python
trainer = trlx.train('gpt2', reward_fn=lambda samples, **kwargs: [sample.count('cats') for sample in samples])
```

For **reward model** training refer to our [autocrit](https://github.com/CarperAI/autocrit) library.

#### Using a reward-labeled dataset

```python
trainer = trlx.train('EleutherAI/gpt-j-6B', samples=['dolphins', 'geese'], rewards=[1.0, 100.0])
```

#### Using a prompt-completion dataset

```python
trainer = trlx.train('gpt2', samples=[['Question: 1 + 2 Answer:', '3'], ['Question: Solve this equation: ∀n>0, s=2, sum(n ** -s). Answer:', '(pi ** 2)/ 6']])
```

#### Trainers provide a wrapper over their underlying model

```python
trainer.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)
```

#### Configure Hyperparameters

```python
from trlx.data.default_configs import default_ppo_config

config = default_ppo_config()
config.model.model_path = 'EleutherAI/gpt-neox-20b'
config.tokenizer.tokenizer_path = 'EleutherAI/gpt-neox-20b'
config.train.seq_length = 2048

trainer = trlx.train(config=config, reward_fn=lambda samples, **kwargs: [len(sample) for sample in samples])
```
To reduce memory usage (if you're experiencing CUDA Out of Memory errors), first try the lowest setting for the following hyperparameters and eventually increase them:
```python
# micro batch size per gpu
config.train.batch_size = 1
# freeze all transformer layers
config.model.num_layers_unfrozen = 0
# maximum sample length, prompts or samples longer than that will be truncated
config.train.seq_length = 128

# micro batch size for sampling (specific for PPO)
config.method.chunk_size = 1
# use an additional Q-head (specific for ILQL)
config.method.two_qs = False
```

#### Save the resulting model to a Hugging Face pretrained language model. (Ready to upload to the Hub!)

```python
trainer.save_pretrained('/path/to/output/folder/')
```

#### Use 🤗 Accelerate to launch distributed training

```bash
accelerate config # choose DeepSpeed option
accelerate launch examples/simulacra.py
```

#### Use NeMo-Megatron to launch distributed training

Follow the setup instructions in the [NeMo README](./trlx/models/).

```bash
python examples/nemo_ilql_sentiments.py
```

For more usage see the [NeMo README](./trlx/models)

#### Use Ray Tune to launch hyperparameter sweep

```bash
ray start --head --port=6379
python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml --accelerate_config configs/accelerate/ddp.yaml --num_gpus 4 examples/ppo_sentiments.py
```

#### Benchmark your trlX fork against trlX's `main` branch
```bash
python -m trlx.reference octocat/trlx-fork:fix-branch
```

## Logging

trlX uses the standard Python `logging` library to log training information to the console. The default logger is set to the `INFO` level, which means that `INFO`, `WARNING`, `ERROR`, and `CRITICAL` level messages will be printed to standard output.

To change the log level directly, you can use the verbosity setter. For example, to set the log level to `WARNING` use:

```python
import trlx

trlx.logging.set_verbosity(trlx.logging.WARNING)
```

This will suppress `INFO` level messages, but still print `WARNING`, `ERROR`, and `CRITICAL` level messages.

You can also control logging verbosity by setting the `TRLX_VERBOSITY` environment variable to one of the standard logging [level names](https://docs.python.org/3/library/logging.html#logging-levels):

- `CRITICAL` (`trlx.logging.CRITICAL`)
- `ERROR` (`trlx.logging.ERROR`)
- `WARNING` (`trlx.logging.WARNING`)
- `INFO` (`trlx.logging.INFO`)
- `DEBUG` (`trlx.logging.DEBUG`)

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

## Citing trlX

```
@inproceedings{havrilla-etal-2023-trlx,
    title = "trl{X}: A Framework for Large Scale Reinforcement Learning from Human Feedback",
    author = "Havrilla, Alexander  and
      Zhuravinskyi, Maksym  and
      Phung, Duy  and
      Tiwari, Aman  and
      Tow, Jonathan  and
      Biderman, Stella  and
      Anthony, Quentin  and
      Castricato, Louis",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.530",
    doi = "10.18653/v1/2023.emnlp-main.530",
    pages = "8578--8595",
}
```

## Acknowledgements

Many thanks to Leandro von Werra for contributing with [trl](https://github.com/lvwerra/trl/), a library that initially inspired this repo.

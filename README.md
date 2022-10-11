[docs-image]: https://readthedocs.org/projects/trlX/badge/?version=latest
[docs-url]: https://trlX.readthedocs.io/en/latest/?badge=latest

# Welcome to Transformer Reinforcement Learning X (`trlX`)
> A repo for distributed training of language models with Reinforcement Learning via Human Feedback (RLHF)

[![Docs Status][docs-image]][docs-url]

**[Documentation](https://trlX.readthedocs.io)**

## Overview
Inspired by the popular `trl` library, the `trlX` repo allows you to fine-tune Huggingface supported language models up to 20B parameters via either reinforcement learning using a provided scoring function or reward-labeled dataset. We aim to support a range of both online and offline RL algorithms including Proximal Policy Optimization (PPO), Natural Language Policy Optimization (NLPO), Actor Critic (A2C), and Implicit Q Learning (ILQL).

The library supports `gpt2` and `gptj` with plans to include `GPT-NeoX`, `T5` and more. PPO and ILQL algorithms are implemented. Disibtributed training has been implemented via HF Accelerate and tested up to two nodes, each with 8 gpus.

## Structure

The training pipeline is broken into four pieces:

- Prompt pipeline: Handles loading of prompts/text used to prompt model for exploration in online methods
- Rollout pipeline: Handles loading and storage of reward labeled data used
- Orchestrator: Handles exploration/rollout collection of online methods. Pushes collected rollouts to the rollout pipeline.
- Model: Wraps the supplied base model (ex: `gpt2`) and implements the desired training method loss (ex: PPO).

Adding a task for RLHF training depends on the desired training method and pre-existing data. If we are online and have no reward labeled data this is as simple as writing a new prompt pipeline, which supplies prompts for exploration, and a new reward function to be passed into the `PPOOrchestrator` class. 

## Example: How to add a task

In the below we implement a sentiment learning task.

### Installation

Install the repo:
```bash
python setup.py develop
```

Configure accelerate:
```bash
accelerate config
```

### Implement a prompt pipeline

```python
@register_datapipeline
class PPOPipeline(BasePipeline):
    def __init__(self, tokenizer, config, prompt_dataset_path = None):
        super().__init__()

        ds = load_dataset('imdb', split='test')
        ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
        ds = ds.filter(lambda x: len(x["review"])<500, batched=False)

        self.tokens = [tokenizer(text,
                                    truncation = True,
                                    padding = 'max_length',
                                    max_length = config.train.input_size,
                                    return_tensors = "pt"
                                 )['input_ids'].long().flatten() for text in ds['review']]
        self.text = [tokenizer.decode(tokens.tolist()) for tokens in self.tokens]

    def __getitem__(self, index : int) -> PromptElement:
        return PromptElement(self.text[index], self.tokens[index])

    def __len__(self) -> int:
        return len(self.text)

    def create_loader(self, batch_size : int, shuffle : bool, prep_fn : Callable = None, num_workers : int = 0) -> DataLoader:
        #TODO(dahoas): Decide how to support varying sizes of prompts without having to tokenize on fly
        def collate_fn(elems : Iterable[PromptElement]) -> PromptElement:
            return PromptBatch(
                [elem.text for elem in elems], torch.stack([elem.tokens for elem in elems])  # Assumes token tensors all same size
            )

        return DataLoader(self, batch_size, shuffle, collate_fn = collate_fn, num_workers = num_workers)
 ```

### Launch training

```python
from typing import List

import torch
from transformers import pipeline

import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")

    sentiment_pipe = pipeline(
        "sentiment-analysis", "lvwerra/distilbert-imdb", device=-1
    )

    def reward_fn(samples: List[str]):
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": None,
            "batch_size": cfg.method.chunk_size,
        }
        pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
        scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
        return scores

    model: AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    if model.accelerator.is_main_process:
        wandb.watch(model.model)

    pipeline: PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch: PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(
        model, pipeline, reward_fn=reward_fn, chunk_size=cfg.method.chunk_size
    )
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
```

And run `accelerate launch my_script.py`

## References

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Language models
The language models utilize the `transformers` library by 🤗 Hugging Face.

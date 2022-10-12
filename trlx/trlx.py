import os
from functools import partial

from numpy.random import RandomState

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.pipeline.offline_pipeline import OfflinePipeline


def train(
    samples, ratings, logit_mask=None, config=None, eval_prompts=[], metric_fn=None
):
    if len(samples) != len(ratings):
        raise ValueError(
            f"Number of samples {len(samples)} should match the number of ratings {len(ratings)}"
        )

    if config is None:
        config = TRLConfig.load_yaml("configs/ilql_config.yml")

    model = AccelerateILQLModel(config=config, logit_mask=logit_mask)

    if model.tokenizer:
        eval_prompts = list(map(model.tokenizer, eval_prompts))

    if len(eval_prompts) == 0 or int(os.environ.get("WORLD_SIZE", 1)) > 1:
        n_eval_prompts = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    else:
        n_eval_prompts = len(eval_prompts)

    # make ad-hoc validation split in case the number of prompts isn't divisible by num_processes
    if len(eval_prompts) < n_eval_prompts and isinstance(eval_prompts, list):
        RandomState(1000).shuffle(samples)
        RandomState(1000).shuffle(ratings)
        ix = n_eval_prompts - len(eval_prompts)
        val_prompts = samples[-ix:]
        samples = samples[:-ix]
        ratings = ratings[:-ix]

        trim_tokenize = partial(
            model.tokenizer, truncation=True, max_length=config.train.input_size
        )
        val_prompts = list(map(trim_tokenize, val_prompts))
        eval_prompts.extend(val_prompts)

    orch = OfflineOrchestrator(
        model, train_samples=samples, ratings=ratings, metric_fn=metric_fn
    )
    model.eval_pipeline = OfflinePipeline(model.tokenizer, eval_prompts)

    model.learn()

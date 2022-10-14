import os
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.pipeline.offline_pipeline import OfflinePipeline

from typing import Tuple, List, Callable, Union, Optional

def train(
        samples: List[str],
        rewards: Optional[List[float]] = None,
        reward_fn: Optional[Callable] = None,
        eval_prompts: Optional[List[str]] = [],
        metric_fn=None,
        config=None,
        logit_mask=None,
        split_token=None,
):
    if reward_fn is not None:
        pass

    elif rewards is not None:
        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")

        model = AccelerateILQLModel(config=config, logit_mask=logit_mask, metric_fn=metric_fn)
        orch = OfflineOrchestrator(model, split_token=split_token)
        model.train_store = orch.make_experience(samples, rewards)
        model.eval_pipeline = OfflinePipeline(eval_prompts, model.tokenizer)

        model.learn()
    else:
        raise ValueError(f"Either {rewards=} or {reward_fn=} should be given")

    return model

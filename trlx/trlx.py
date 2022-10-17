import os

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.pipeline.offline_pipeline import PromptPipeline

from typing import Tuple, List, Callable, Union, Optional

def train(
        samples: Optional[List[str]] = None,
        rewards: Optional[List[float]] = None,
        reward_fn: Optional[Callable] = None,
        prompts: Optional[List[str]] = None,
        eval_prompts: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        metric_fn: Optional[Callable] = None,
        config=None,
        logit_mask=None,
        split_token=None,
):
    """
    Dispatches online or offline reinforcement training depending on whether a reward function or a list of samples & rewards is given

    Args:
        reward_fn (Iterable[str] -> Iterable[float]): Function to rate batches of generated samples
        samples (Iterable[str]): List of samples from an exisiting dataset
        rewards (Iterable[float]): Rewards, ratings or labels for provided samples
    """
    if reward_fn is not None:
        if config is None:
            config = TRLConfig.load_yaml("configs/ppo_config.yml")

        if model_path:
            config.model.model_path = model_path

        if eval_prompts is None:
            eval_prompts = prompts[:config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))]

        model = get_model(config.model.model_type)(config)
        pipeline = PromptPipeline(prompts, model.tokenizer)
        model.eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)

        orch: PPOOrchestrator = get_orchestrator(config.train.orchestrator)(
            model, pipeline, reward_fn=reward_fn, chunk_size=config.method.chunk_size
        )
        orch.make_experience(config.method.num_rollouts)

    elif rewards is not None:
        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")

        if model_path:
            config.model.model_path = model_path

        model = AccelerateILQLModel(config=config, logit_mask=logit_mask, metric_fn=metric_fn)
        orch = OfflineOrchestrator(model, split_token=split_token)
        model.train_store = orch.make_experience(samples, rewards)
        model.eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)

    else:
        raise ValueError(f"Either {rewards=} or {reward_fn=} should be given")

    model.learn()
    return model

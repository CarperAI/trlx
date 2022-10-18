import os
from typing import Callable, Iterable, List, Optional, Tuple

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.utils.loading import get_model, get_orchestrator


def train(
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    config=None,
    logit_mask=None,
    split_token=None,
):
    """
    Dispatches online or offline reinforcement training depending on whether a reward function or a list of samples & rewards is given

    Args:
        model_path (Optional[str]): Path to huggingface checkpoint or local directory path
        reward_fn (Iterable[str] -> Iterable[float]): Function to rate batches of generated samples
        dataset (Iterable[str], Iterable[float]): List of samples and rewards
    """

    if reward_fn is not None:
        if config is None:
            config = TRLConfig.load_yaml("configs/ppo_config.yml")

        if model_path:
            config.model.model_path = model_path

        if eval_prompts is None:
            eval_prompts = prompts[
                : config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
            ]

        model = get_model(config.model.model_type)(config)

        pipeline = PromptPipeline(prompts, model.tokenizer)
        orch: PPOOrchestrator = get_orchestrator(config.train.orchestrator)(
            model, pipeline, reward_fn=reward_fn, chunk_size=config.method.chunk_size
        )
        orch.make_experience(config.method.num_rollouts)
        eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)
        model.add_eval_pipeline(eval_pipeline)

    elif dataset is not None:
        samples, rewards = dataset

        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")

        if model_path:
            config.model.model_path = model_path

        model = AccelerateILQLModel(
            config=config, logit_mask=logit_mask, metric_fn=metric_fn
        )

        if eval_prompts is None:
            eval_prompts = (
                [model.tokenizer.bos_token]
                * config.train.batch_size
                * int(os.environ.get("WORLD_SIZE", 1))
            )

        eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)
        orch = OfflineOrchestrator(model, split_token=split_token)
        orch.make_experience(samples, rewards)
        model.add_eval_pipeline(eval_pipeline)

    else:
        raise ValueError(f"Either {dataset=} or {reward_fn=} should be given")

    model.learn()
    return model

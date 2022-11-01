import os
from typing import Callable, Iterable, List, Optional, Tuple

from accelerate import Accelerator

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import AccelerateILQLModel
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.utils.loading import get_model, get_orchestrator

import ray


def train(
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[Callable] = None,
    config: Optional[TRLConfig] = None,
    split_token: Optional[str] = None,
    logit_mask: Optional[List[List[bool]]] = None,
):
    """
    Dispatches online or offline reinforcement training depending on whether a reward function or a list of samples & rewards is given

    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        reward_fn (List[str] -> List[float]): Function to rate batches of generated samples
        dataset (List[str], List[float]): Lists of samples and rewards
        prompts (List[str]): Prompts to sample off from during online training
        eval_prompts (List[str]): Prompts to periodically validate training on
        metric_fn (Optional[Callable[List[str], List[float]]]): Function to compute statistics on validation samples
        split_token (Optional[str]): Split samples in the dataset on prompts and continuations
        logit_mask (Optional[List]): Bigram masking matrix
    """

    if reward_fn is not None:
        config_path = "configs/ppo_config.yml"
    elif dataset is not None:
        config_path = "configs/ilql_config.yml"

    if config is None:
        config = TRLConfig.load_yaml(config_path)

    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb")

    # Initialize tracker
    if accelerator.is_main_process and not ray.is_initialized():
        accelerator.init_trackers(
            project_name=config.train.project_name,
            config=config.to_dict(),
            init_kwargs={
                "wandb": {
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }
            },
        )

        run = accelerator.get_tracker("wandb")

    if reward_fn is not None:
        if model_path:
            config.model.model_path = model_path

        model: AcceleratePPOModel = get_model(config.model.model_type)(
            config, accelerator
        )

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        prompts = prompts or [model.tokenizer.bos_token] * batch_size

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

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

        if model_path:
            config.model.model_path = model_path

        model = AccelerateILQLModel(
            config=config,
            accelerator=accelerator,
            logit_mask=logit_mask,
            metric_fn=metric_fn,
        )

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        if eval_prompts is None:
            eval_prompts = [model.tokenizer.bos_token] * batch_size

        eval_pipeline = PromptPipeline(eval_prompts, model.tokenizer)

        orch = OfflineOrchestrator(model, split_token=split_token)
        orch.make_experience(samples, rewards)
        model.add_eval_pipeline(eval_pipeline)

    else:
        raise ValueError(f"Either {dataset=} or {reward_fn=} should be given")

    model.learn()

    if accelerator.is_main_process and not ray.is_initialized():
        accelerator.end_training()

    return model

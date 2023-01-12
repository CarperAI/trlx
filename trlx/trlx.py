import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from trlx.data.configs import TRLConfig
from trlx.utils import set_seed
from trlx.utils.loading import get_orchestrator, get_pipeline, get_trainer


def train(
    model_path: Optional[str] = None,
    reward_fn: Optional[
        Callable[[List[str], List[str], List[str]], List[float]]
    ] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[
        Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]
    ] = None,
    config: Optional[TRLConfig] = None,
    logit_mask: Optional[List[List[bool]]] = None,
    stop_word: Optional[str] = "",
):
    """
    Dispatches online or offline reinforcement training
    depending on whether a reward function or a list of samples & rewards is given

    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        reward_fn (Optional[Callable[[List[str], List[str], List[str]], List[float]]]):
            Function to rate batches of generated samples. Its arguments are
            (`samples`, `prompts`, `outputs`) and the return is a list of `rewards` per each sample
        dataset (List[Union[str, List[str]]], List[float]):
            Lists of samples and rewards for offline training. Samples consist of a variable number
            of prompts (questions, environment states etc.) and outputs which are meant to be optimized.
            Following form is expected (prompt_0: str, output_0: str, prompt_1: str, output_1: str ...).
            Giving a single string `s` for the sample is a shorthand for (`tokenizer.bos_token`, `s`)
        prompts (List[str]): Prompts to sample off from during online training
        eval_prompts (List[str]): Prompts to periodically validate training on
        metric_fn (Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]]):
            Function to compute statistics on batches of gnerated samples. Its arguments are the same
            as in `reward_fn` (`samples`, `prompts`, `outputs`) but the return is dictionary with keys
            as metric's name and values and lists of numeric values per each sample in batch
        config (Optional[TRLConfig]): TRL configuration object to override default settings
        logit_mask (Optional[List]): Bigram masking matrix
        stop_word (Optional[str]): A string to trim generations (either for experience or evaluation) up to
            its encounter in them
    """

    if reward_fn is not None:
        if config is None:
            config = TRLConfig.load_yaml("configs/ppo_config.yml")
        set_seed(config.train.seed)

        if model_path:
            config.model.model_path = model_path

        trainer = get_trainer(config.train.trainer)(
            config=config, reward_fn=reward_fn, metric_fn=metric_fn, stop_word=stop_word
        )

        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        prompts = prompts or [trainer.tokenizer.bos_token] * batch_size

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        max_prompt_length = (
            config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
        )
        pipeline = get_pipeline(config.train.pipeline)(
            prompts, max_prompt_length, trainer.tokenizer
        )
        orch = get_orchestrator(config.train.orchestrator)(
            trainer, pipeline, chunk_size=config.method.chunk_size
        )
        orch.make_experience(config.method.num_rollouts)

        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts, max_prompt_length, trainer.tokenizer
        )
        trainer.add_eval_pipeline(eval_pipeline)

    elif dataset is not None:
        samples, rewards = dataset

        if len(samples) != len(rewards):
            raise ValueError(
                f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}"
            )

        if config is None:
            config = TRLConfig.load_yaml("configs/ilql_config.yml")
        set_seed(config.train.seed)

        if model_path:
            config.model.model_path = model_path

        trainer = get_trainer(config.train.trainer)(
            config=config,
            metric_fn=metric_fn,
            stop_word=stop_word,
            logit_mask=logit_mask,
        )
        batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        max_prompt_length = (
            config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
        )

        if eval_prompts is None:
            eval_prompts = [trainer.tokenizer.bos_token] * batch_size
        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts, max_prompt_length, trainer.tokenizer
        )

        orch = get_orchestrator(config.train.orchestrator)(trainer)
        orch.make_experience(samples, rewards, config.train.seq_length)
        trainer.add_eval_pipeline(eval_pipeline)

    else:
        raise ValueError(f"Either {dataset=} or {reward_fn=} should be given")

    trainer.learn()
    return trainer

import os
import tempfile
import unittest
from typing import List, Mapping
from unittest.mock import patch

import trlx.utils.logging as logging
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
from trlx.utils.loading import get_pipeline, get_trainer

logging.disable_progress_bar()
logging.set_verbosity(logging.ERROR)


def get_default_train_and_eval_prompts() -> Mapping[str, List[str]]:
    return dict(
        train=[
            "The quick brown fox jumps over the lazy",
            "The cat sat on the mat next to the",
            "What sort of food does a",
            "The nextdoor neighbor's fence couldn't keep the",
            "When Tom got home from work he had to walk his",
        ],
        eval=[
            "I purchased a collar for my new",
            "I couldn't help but laugh when the mailman was chased by the",
        ],
    )


def get_default_reward_fn():
    def reward_fn(samples: List[str], **kwargs):
        return [sample.count("dog") for sample in samples]

    return reward_fn


class TestAccelerateBaseTrainer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.prompt_dataset = get_default_train_and_eval_prompts()

    @classmethod
    def get_default_config(cls):
        return TRLConfig(
            train=TrainConfig(
                seq_length=16,
                epochs=1,
                total_steps=8,
                batch_size=2,
                checkpoint_interval=4,
                checkpoint_dir="checkpoints",
                eval_interval=8,
                pipeline="PromptPipeline",
                trainer="AcceleratePPOTrainer",
                tracker=None,
            ),
            model=ModelConfig(model_path="gpt2", num_layers_unfrozen=2),
            tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
            optimizer=OptimizerConfig(
                name="adamw", kwargs=dict(lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
            ),
            scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-4)),
            method=PPOConfig(
                name="PPOConfig",
                num_rollouts=128,
                chunk_size=128,
                ppo_epochs=4,
                init_kl_coef=0.05,
                target=6,
                horizon=10000,
                gamma=1,
                lam=0.95,
                cliprange=0.2,
                cliprange_value=0.2,
                vf_coef=1,
                scale_reward="ignored",
                ref_mean=None,
                ref_std=None,
                cliprange_reward=10,
                gen_kwargs=dict(
                    max_new_tokens=6,
                    top_k=0,
                    top_p=1.0,
                    do_sample=True,
                ),
            ),
        )

    def get_trainer(self, config: TRLConfig):
        trainer = get_trainer(config.train.trainer)(
            config=config,
            reward_fn=get_default_reward_fn(),
            metric_fn=None,
            stop_sequences=None,
            **config.train.trainer_kwargs,
        )

        max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
        train_pipeline = get_pipeline(config.train.pipeline)(
            self.prompt_dataset["train"], max_prompt_length, trainer.tokenizer
        )
        trainer.add_prompt_pipeline(train_pipeline)
        trainer.make_experience(config.method.num_rollouts)

        eval_pipeline = get_pipeline(config.train.pipeline)(
            self.prompt_dataset["eval"], max_prompt_length, trainer.tokenizer
        )
        trainer.add_eval_pipeline(eval_pipeline)
        return trainer

    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.get_default_config()
            config.train.checkpoint_dir = tmpdir

            trainer = self.get_trainer(config)
            trainer.learn()

            total_steps = config.train.total_steps
            interval = config.train.checkpoint_interval
            for i in range(interval, total_steps + 1, interval):
                checkpoint_dir = os.path.join(tmpdir, f"checkpoint_{i}")
                self.assertTrue(os.path.isdir(checkpoint_dir))
            if total_steps % interval != 0:
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, f"checkpoint_{total_steps}")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "best_checkpoint")))

    def test_accumulate_context(self):
        config = self.get_default_config()
        trainer = self.get_trainer(config)
        trainer.accelerator.gradient_accumulation_steps = 3

        def run_test(mb_count, num_mb, total_steps, should_call_no_sync):
            trainer.mb_count = mb_count
            trainer.num_mb = num_mb
            trainer.config.train.total_steps = total_steps

            with patch.object(trainer.accelerator, "no_sync") as no_sync_tracker:
                with patch("contextlib.nullcontext") as nullcontext_tracker:
                    with trainer._accumulate():
                        pass

            self.assertEqual(no_sync_tracker.called, should_call_no_sync)
            self.assertEqual(nullcontext_tracker.called, not should_call_no_sync)

        # Test case 1: the context manager should call accelerator.no_sync
        run_test(mb_count=1, num_mb=2, total_steps=4, should_call_no_sync=True)

        # Test case 2: the context manager should sync because next mb_count is 3 (corresponds with gradient accumulation)
        run_test(mb_count=2, num_mb=2, total_steps=4, should_call_no_sync=False)

        # Test case 3: the context manager should sync because next mb_count is final step even though it is not % by 3
        run_test(mb_count=3, num_mb=1, total_steps=4, should_call_no_sync=False)

        # Test case 4: the context manager should call accelerator.no_sync
        run_test(mb_count=3, num_mb=1, total_steps=6, should_call_no_sync=True)

        # Test case 5: the context manager should sync because next mb_count is 28 and 28 // num_mb means it is the last step
        run_test(mb_count=27, num_mb=4, total_steps=7, should_call_no_sync=False)

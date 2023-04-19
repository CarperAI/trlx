import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=10,
        epochs=20,
        total_steps=10000,
        batch_size=100,
        checkpoint_interval=10000,
        eval_interval=20,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(model_path="CarperAI/randomwalks", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="CarperAI/randomwalks", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=3.0e-4)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=128,
        ppo_epochs=4,
        init_kl_coef=0,
        target=None,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.2,
        scale_reward="ignored",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=1,
        gen_kwargs=dict(
            max_new_tokens=9,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    metric_fn, prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        # An "optimality" reward function is used, with scores in [0,1]
        # depending on how close the path is to the shortest possible path.
        reward_fn=lambda samples, prompts, outputs: metric_fn(samples)["optimality"],
        # The prompts are simply the first nodes (represented as letters) to
        # start from.
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=lambda samples, prompts, outputs: metric_fn(samples),
        config=config,
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

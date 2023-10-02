import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.trainer.accelerate_rft_trainer import RFTConfig

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=10,
        epochs=100,
        total_steps=1000,
        batch_size=100,
        checkpoint_interval=1000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateRFTTrainer",
        checkpoint_dir="checkpoints/randomwalks",
    ),
    model=ModelConfig(model_path="CarperAI/randomwalks", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="CarperAI/randomwalks", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=3.0e-4, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=3.0e-4)),
    method=RFTConfig(
        name="RFTConfig",
        n_generations_per_prompt=100,
        start_percentile=0.9,
        end_percentile=0.95,
        n_improve_steps=1,
        gen_kwargs=dict(
            max_new_tokens=9,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            do_sample=True,
        ),
    ),
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    metric_fn, prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        reward_fn=lambda samples, **kwargs: metric_fn(samples)["optimality"],
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=lambda samples, **kwargs: metric_fn(samples),
        config=config,
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

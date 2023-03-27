from transformers import GPT2Config

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def main(hparams):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, eval_prompts, walks, _ = generate_random_walks(seed=config.train.seed)
    rewards = metric_fn(walks)["optimality"]
    # split each random walk into (starting state, rest of the walk)
    walks = [[walk[:1], walk[1:]] for walk in walks]

    trlx.train(
        model_path=GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        samples=walks,
        rewards=rewards,
        eval_prompts=eval_prompts,
        metric_fn=lambda samples, **kwargs: metric_fn(samples),
        config=config,
        stop_sequences=["|"],
    )


default_config = TRLConfig(
    train=TrainConfig(
        seq_length=11,
        batch_size=100,
        epochs=20,
        total_steps=1000,
        checkpoint_interval=1000,
        eval_interval=16,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
    ),
    model=ModelConfig(model_path=GPT2Config(n_layer=6, n_embd=144, vocab_size=23), num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="CarperAI/randomwalks", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=2e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1000, eta_min=2e-4)),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.8,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.1,
        beta=0,
        steps_for_target_q_sync=5,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=9, top_k=10, beta=[0, 1, 100], temperature=1.0),
    ),
)

if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)

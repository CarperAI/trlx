from .configs import TRLConfig, TrainConfig, ModelConfig, TokenizerConfig, OptimizerConfig, SchedulerConfig
from ..trainer.nn.ppo_models import PPOConfig
from ..trainer.nn.ilql_models import ILQLConfig

def default_ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer"
        ),
        model=ModelConfig(
            model_path="lwerra/gpt2-imdb",
            num_layers_unfrozen=2
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="gpt2",
            truncation_side="right"
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(
                lr=1.0e-4,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=1.0e-6
            )
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs=dict(
                T_max=10000,
                eta_min=1.0e-4
            )
        ),
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
                max_new_tokens=40,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            )
        )
    )
'''
train:
  seq_length: 64
  batch_size: 128
  epochs: 100
  total_steps: 1000

  checkpoint_interval: 1000
  eval_interval: 100

  pipeline: "PromptPipeline"
  trainer: "AccelerateILQLTrainer"
  seed: 1000

model:
  model_path: "gpt2"
  num_layers_unfrozen: -1

tokenizer:
  tokenizer_path: "gpt2"
  truncation_side: "right"

optimizer:
  name: "adamw"
  kwargs:
    lr: 5.0e-5
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 1.0e-6

scheduler:
  name: "cosine_annealing"
  kwargs:
    T_max: 1000 # train.total_steps
    eta_min: 5.0e-5

method:
  name: "ilqlconfig"
  tau: 0.7
  gamma: 0.99
  cql_scale: 0.1
  awac_scale: 1
  alpha: 0.001
  beta: 0
  steps_for_target_q_sync: 5
  two_qs: true
  gen_kwargs:
    max_new_tokens: 56
    top_k: 20
    beta: 4
    temperature: 1.0
'''
# rewrite the above in python below

def default_ilql_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=64,
            batch_size=128,
            epochs=100,
            total_steps=1000,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateILQLTrainer"
        ),
        model=ModelConfig(
            model_path="gpt2",
            num_layers_unfrozen=-1
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path="gpt2",
            truncation_side="right"
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(
                lr=5.0e-5,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=1.0e-6
            )
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs=dict(
                T_max=1000, # train.total_steps
                eta_min=5.0e-5
            )
        ),
        method=ILQLConfig(
            name="ilqlconfig",
            tau=0.7,
            gamma=0.99,
            cql_scale=0.1,
            awac_scale=1,
            alpha=0.001,
            beta=0,
            steps_for_target_q_sync=5,
            two_qs=True,
            gen_kwargs=dict(
                max_new_tokens=56,
                top_k=20,
                beta=4,
                temperature=1.0
            )
        )
    )



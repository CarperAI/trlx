.. _configs:

Configs
************************

Training requires configuration to be passed through a set of configs: `TrainConfig` with training configuration, `ModelConfig`, `TokenizerConfig`, `OptimizerConfig`, `SchedulerConfig` and `MethodConfig` for a specific configuration of the algorithm (PPO, ILQL or SFT)

**General**

.. autoclass:: trlx.data.configs.TRLConfig
    :members:

.. autoclass:: trlx.data.configs.TrainConfig
    :members:

.. autoclass:: trlx.data.configs.ModelConfig
    :members:

.. autoclass:: trlx.data.configs.TokenizerConfig
    :members:

.. autoclass:: trlx.data.configs.OptimizerConfig
    :members:

.. autoclass:: trlx.data.configs.SchedulerConfig
    :members:

.. autoclass:: trlx.data.method_configs.MethodConfig
    :members:

**PPO**

.. autoclass:: trlx.models.modeling_ppo.PPOConfig
    :members:

**ILQL**

.. autoclass:: trlx.models.modeling_ilql.ILQLConfig
    :members:

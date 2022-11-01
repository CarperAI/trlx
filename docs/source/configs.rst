.. _configs:

Configs
************************

Training a model in TRL will require you to set several configs:
ModelConfig, which contains general info on the model being trained. TrainConfig, which contains things like
training hyperparameters. And finally, MethodConfig, which contains hyperparameters or settings for
the specific method being used (i.e. ILQL or PPO)


**General**

.. autoclass:: trlx.data.configs.TRLConfig
    :members:

.. autoclass:: trlx.data.configs.ModelConfig
    :members:

.. autoclass:: trlx.data.configs.TrainConfig
    :members:

.. autoclass:: trlx.data.method_configs.MethodConfig
    :members:

**PPO**

.. autoclass:: trlx.data.method_configs.PPOConfig
    :members:

**ILQL**

.. autoclass:: trlx.data.method_configs.ILQLConfig
    :members:

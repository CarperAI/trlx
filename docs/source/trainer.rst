.. _trainers:

RL Trainers
*******************

RL Trainers are what you're training with trlX. Currently, we support PPO and ILQL.
Note that new trainers must be registered with ``trlx.trainer.register_trainer``.

**General**

.. autoclass:: trlx.trainer.BaseRLTrainer
    :members:

.. autoclass:: trlx.trainer.accelerate_base_trainer.AccelerateRLTrainer
    :members:

**PPO**

.. autoclass:: trlx.trainer.accelerate_ppo_trainer.AcceleratePPOTrainer
    :members:

.. autoclass:: trlx.trainer.nn.ppo_models.CausalLMWithValueHead
    :members:

.. autoclass:: trlx.trainer.nn.ppo_models.GPTModelBranch
    :members:

.. autoclass:: trlx.trainer.nn.ppo_models.OPTModelBranch
    :members:

.. autoclass:: trlx.trainer.nn.ppo_models.CausalLMHydraWithValueHead
    :members:

**ILQL**

.. autoclass:: trlx.trainer.accelerate_ilql_trainer.AccelerateILQLTrainer
    :members:

.. autoclass:: trlx.trainer.nn.ilql_models.CausalLMWithValueHeads
    :members:

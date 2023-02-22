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

**ILQL**

.. autoclass:: trlx.trainer.accelerate_ilql_trainer.AccelerateILQLTrainer
    :members:

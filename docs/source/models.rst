.. _models:

RL Models
*******************

RL Models are what you're training with trlX. Currently, we support PPO and ILQL.
Note that new models must be registered with ``trlx.model.register_model``.

**General**

.. autoclass:: trlx.model.BaseRLModel
    :members:

.. autoclass:: trlx.model.accelerate_base_model.AccelerateRLModel
    :members:

**PPO**  

.. autoclass:: trlx.model.accelerate_ppo_model.AcceleratePPOModel
    :members:

.. autoclass:: trlx.model.nn.ppo_models.GPTHeadWithValueModel
    :members:

.. autoclass:: trlx.model.nn.ppo_models.ModelBranch
    :members:

.. autoclass:: trlx.model.nn.ppo_models.GPTHydraHeadWithValueModel
    :members:

**ILQL**

.. autoclass:: trlx.model.accelerate_ilql_model.AccelerateILQLModel
    :members:

.. autoclass:: trlx.model.nn.ilql_models.CausalLMWithValueHeads
    :members:
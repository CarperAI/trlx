.. _data:

Data Elements
************************

All of the major Carper projects: trlX, CHEESE, and magiCARP use
dataclasses corresponding to batches of data to communicate data between models and different
components. trlX is no different, though it has many different dataclasses for
different components like training or inference. Currently, we support PPO and ILQL, which
each demand different kinds of data during training.


**Basic Data Elements for Accelerate**

.. autoclass:: trlx.data.accelerate_base_datatypes.PromptElement
    :members:

.. autoclass:: trlx.data.accelerate_base_datatypes.PromptBatch
    :members:

.. autoclass:: trlx.data.accelerate_base_datatypes.AccelerateRLElement
    :members:

.. autoclass:: trlx.data.accelerate_base_datatypes.AccelerateRLBatchElement
    :members:

**Data Elements for PPO**

.. autoclass:: trlx.data.ppo_types.PPORLElement
    :members:

.. autoclass:: trlx.data.ppo_types.PPORLBatch
    :members:

**Data Elements for ILQL**

.. autoclass:: trlx.data.ilql_types.ILQLElement
    :members:

.. autoclass:: trlx.data.ilql_types.ILQLBatch
    :members:

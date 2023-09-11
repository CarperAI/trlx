.. _pipeline:

Pipelines
************************

Pipeline are used to store the data in appropriate format used for training models

**General**

.. autoclass:: trlx.pipeline.BasePipeline
    :members:

.. autoclass:: trlx.pipeline.BaseRolloutStore
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.DialogMessage
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.DialogStore
    :members:

.. autofunction:: trlx.pipeline.offline_pipeline.tokenize_dialogue

**PPO**

.. autoclass:: trlx.pipeline.ppo_pipeline.PPORolloutStorage
    :members:

**ILQL**

.. autoclass:: trlx.pipeline.offline_pipeline.PromptPipeline
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.ILQLRolloutStorage
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.ILQLSeq2SeqRolloutStorage
    :members:

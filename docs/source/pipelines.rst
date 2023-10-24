.. _pipeline:

Pipelines
=========

Pipelines are used for accumulation and convertion of the training data to appropriate format.

.. autoclass:: trlx.pipeline.BasePipeline
    :members:

.. autoclass:: trlx.pipeline.BaseRolloutStore
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.DialogMessage
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.DialogStore
    :members:

.. autofunction:: trlx.pipeline.offline_pipeline.tokenize_dialogue

.. autoclass:: trlx.pipeline.ppo_pipeline.PPORolloutStorage
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.PromptPipeline
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.ILQLRolloutStorage
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.ILQLSeq2SeqRolloutStorage
    :members:

.. _pipeline:

Pipelines and Rollout Store
***************************

*Pipelines*

Pipelines in trlX provide a way to read from a dataset. They are used to fetch data from the dataset and feed it to the models for training or inference. The pipelines allow for efficient processing of the data and ensure that the models have access to the data they need for their tasks.

.. autoclass:: trlx.pipeline.BasePipeline
    :members:

.. autoclass:: trlx.pipeline.BaseRolloutStore
    :members:


*Rollout Stores*

Rollout stores in trlX are used to store experiences created for the models by the orchestrator. The experiences in the rollout stores serve as the training data for the models. The models use the experiences stored in their rollout stores to learn and improve their behavior. The rollout stores provide a convenient and efficient way for the models to access the experiences they need for training.


**PPO**

.. autoclass:: trlx.pipeline.ppo_pipeline.PPORolloutStorage
    :members:

**ILQL**

.. autoclass:: trlx.pipeline.offline_pipeline.PromptPipeline
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.ILQLRolloutStorage
    :members:

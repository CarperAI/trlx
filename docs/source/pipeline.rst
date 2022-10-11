.. _pipeline:

Pipelines
************************

Pipelines are how you read from a dataset with trlX. Rollout stores are how models store experiences created
for them by the orchestrator. It is these experiences in their rollout store that they are trained on.

**General**

.. autoclass:: trlx.pipeline.BasePipeline
    :members:

.. autoclass:: trlx.pipeline.BaseRolloutStore
    :members:

.. autoclass:: trlx.pipeline.accelerate_base_pipeline.AcceleratePipeline
    :members:

.. autoclass:: trlx.pipeline.accelerate_base_pipeline.AccelerateRolloutStorage
    :members:

**PPO**

.. autoclass:: trlx.pipeline.ppo_pipeline.PPOPipeline
    :members:

.. autoclass:: trlx.pipeline.ppo_pipeline.PPORolloutStorage
    :members:

**ILQL**

.. autoclass:: trlx.pipeline.offline_pipeline.OfflinePipeline
    :members:

.. autoclass:: trlx.pipeline.offline_pipeline.OfflineRolloutStorage
    :members:

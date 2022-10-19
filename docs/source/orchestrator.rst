.. _orchestrator:

Orchestrators
*******************

Orchestrators manage reading data from a pipeline and creating RL data elements (i.e. ``trlx.data.RLElement``)
to push to a models rollout storage. Use the ``trlx.orchestrator.register_orchestrator`` decorator when creating
new orchestrators.

**General**

.. autoclass:: trlx.orchestrator.Orchestrator
   :members:

**PPO**

.. autoclass:: trlx.orchestrator.ppo_orchestrator.PPOOrchestrator
    :members:

**ILQL**

.. autoclass:: trlx.orchestrator.offline_orchestrator.OfflineOrchestrator
    :members:

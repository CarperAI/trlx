.. _api:

API
===

trlX uses a single entrypoint for training, which will execute training conditioned on the passed config and the necessary arguments for a specific training routine. For the online training `prompts` (a list of strings to prompt the training model) and `reward_fn` (a function which gives reward for model outputs sampled from `prompts`) are necessary, while for offline training `samples` (a list of environment/model interactions) and `rewards` (precomputed scores for each interaction) are required.

Training
--------

.. autofunction:: trlx.train

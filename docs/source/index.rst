Welcome to trlX's documentation!
================================
trlX is a library for training large language models with reinforcement learning. It supports two RL algorithms, PPO for online training and ILQL for offline training with a collected dataset. For distributed training it can leverage two backends: `Huggingface 🤗 Accelerate <https://github.com/huggingface/accelerate>`_ and `NVIDIA NeMo <https://nvidia.github.io/NeMo>`_. Check out the examples for usage instructions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   examples
   configs
   trainers
   data
   pipelines

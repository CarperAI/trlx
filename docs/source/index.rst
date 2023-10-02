Welcome to trlX's documentation!
================================
trlX is a library for training large language models with reinforcement learning. Training can be done with two RL algorithms: PPO (`Schulman et al. 2017 <https://arxiv.org/abs/1707.06347>`_) for online training and ILQL (`Snell et al. 2022 <https://arxiv.org/abs/2206.11871>`_) for offline training. For distributed training two backends are supported: `Huggingface 🤗 Accelerate <https://github.com/huggingface/accelerate>`_ and `NVIDIA NeMo <https://nvidia.github.io/NeMo>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   examples
   configs
   trainers
   pipelines
   data

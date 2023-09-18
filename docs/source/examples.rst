.. _examples:

Examples
========

Random Walks
------------

This is a simple toy example used for testing described in `Decision Transformer
(Lili Chen et al. 2021) <https://arxiv.org/abs/2106.01345>`_ simple enough that
it doesn't require GPU access.

Description
^^^^^^^^^^^

The task is to find the shortest path on a directed graph. The reward is based
on how optimal the path is compared to the shortest possible. Paths are
represented as strings of letters, with each letter corresponding to a node in
the graph.

Training
^^^^^^^^

For `PPO Training
<https://github.com/CarperAI/trlx/blob/main/examples/randomwalks/ppo_randomwalks.py>`_,
a language model continually samples paths in a graph and directly optimizes for
their shortness using surrogate reward function. For `ILQL Training
<https://github.com/CarperAI/trlx/blob/main/examples/randomwalks/ilql_randomwalks.py>`_
a language model learns directly from a set of 1000 pre-sampled randomwalks in a
graph paired with their relative lengths' shortness.

Positive Sentiment
------------------

Description
^^^^^^^^^^^
The goal is to optimize a language model to generate positive sentiment responses to a given prompt.

Training
^^^^^^^^

The training is done by using `PPO trainer
<https://github.com/CarperAI/trlx/blob/main/examples/ppo_sentiments.py>`_ to
maximize a score from pre-trained sentiment classifier trained on IMDB review
sentiments `dataset <https://huggingface.co/datasets/imdb>`_ . For `ILQL Training
<https://github.com/CarperAI/trlx/blob/main/examples/ilql_sentiments.py>`_ the
model is trained directly on the dataset and its labels: `0` for a negative
review and `1` for a positive one. For `SFT Training
<https://github.com/CarperAI/trlx/blob/main/examples/sft_sentiments.py>`_ the
model is trained only on the positive reviews.


Helpful & Harmless
-------------------

Description
^^^^^^^^^^^

The goal of the training is to improve both helpfulness and harmlessness of the
model's outputs following Anthropic's paper `Training a Helpful and Harmless
Assistant with Reinforcement Learning from Human Feedback
<https://arxiv.org/abs/2204.05862>`_

Training
^^^^^^^^

The training is done by either utilizing a reward model trained on the
Anthropic's Helpful & Harmless `dataset
<https://github.com/anthropics/hh-rlhf>`_ using `PPO trainer
<https://github.com/CarperAI/trlx/blob/main/examples/hh/ppo_hh.py>`_, or by
using the dataset directly by reward labeling each selected and rejected with
`+1` and `-1` respectively using `ILQL trainer
<https://github.com/CarperAI/trlx/blob/main/examples/hh/ilql_hh.py>`_, or using
`SFT trainer
<https://github.com/CarperAI/trlx/blob/main/examples/hh/sft_hh.py>`_ and
finetuning only over selected responses.

The setup used for this example assumes a single machine with 8xA100 80GB, the
last of which will be dedicated to hosting a reward model. Optionally you can
use `Triton Inference Server <https://github.com/triton-inference-server>`_ to
host it elsewhere, otherwise the training script will instantiate it (`a
pretrained one <https://huggingface.co/Dahoas/gptj-rm-static>`_) on its own.

Launch training of `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6B>`_ on 7
GPUs with 8th GPU hosting a reward model:

.. code-block:: console

    accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py
    # or for training from other predefined checkpoint
    CONFIG_NAME=125M accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py

Optional steps to setup a reward model using Triton Server:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    # convert the model and create a config and a folder `model_store` structured for Triton
    python to_triton.py --base_model EleutherAI/gpt-j-6B --checkpoint Dahoas/gptj-rm-static --revision 676bfd4d

    # convert the docker image (skip this if you use docker instead)
    singularity build --sandbox tritonserver-pyt.sif docker://nvcr.io/nvidia/tritonserver:22.08-pyt-python-py3

    # start Triton Server pointing to the `model_store` containing the reward model
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=7 singularity run --nv --bind model_store:/model_store tritonserver-pyt.sif tritonserver --model-repository=/model_store &

Launch training:

.. code-block:: console

     # set model's url and replace the name after the slash if you use a different checkpoint
     export TRITON_HOST=localhost:8001/gptj-rm-static
     accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py

W&B runs:

PPO GPT-J: https://wandb.ai/sorry/trlx/runs/v0bir5s9

ILQL GPT-J: https://wandb.ai/sorry/trlx/runs/1qqxp72a

SFT GPT-J: https://wandb.ai/sorry/trlx/runs/a7ng078v

.. _api:

API
===

trlX uses a single entrypoint for training, which will execute training conditioned on the passed config and the necessary arguments for a specific training routine. For the online training `prompts` (a list of strings to prompt the training model) and `reward_fn` (a function which gives reward for model outputs sampled from `prompts`) are necessary, while for offline training `samples` (a list of environment/model interactions) and `rewards` (precomputed scores for each interaction) are required.

Training
--------

.. autofunction:: trlx.train

Distributed
-----------

Accelerate
^^^^^^^^^^

To launch distributed training with Accelerate, first you have to specify the training configuration. You only have to execute this command once per each training node.

.. code-block:: console

    $ accelerate config
    $ accelerate launch examples/ppo_sentiments.py

You can also use configs provided in `trlX repository <https://github.com/CarperAI/trlx/tree/main/configs/accelerate>`_):

.. code-block:: console

    $ accelerate launch --config_file configs/accelerate/zero2-bf16.yaml examples/ppo_sentiments.py


NVIDIA NeMo
^^^^^^^^^^^

For training with NeMo you have to use a model stored in the NeMo format. You can convert an existing llama model with the following script:

.. code-block:: console

    $ python examples/llama_nemo/convert_llama_to_nemo.py --model_path NousResearch/Llama-2-7b-hf --output_folder nemo_llama2_7b --total_tp 4 --name 7b

To start training you have to execute python script per each GPU, or launch the following sbatch script which has `-ntasks-per-node=8`

.. code-block:: console

    $ sbatch examples/llama_nemo/dist_train.sh

Run example: `wandb <https://wandb.ai/carperai/trlxnemo/runs/v7592y73?workspace=user-pvduy>`_

### Training on Anthropic's Helpful & Harmless [dataset](https://github.com/anthropics/hh-rlhf)

As an example, the following setup assumes a single machine with 8xA100 80GB, the last of which will be dedicated to hosting a reward model. Optionally you can use [Triton Inference Server](https://github.com/triton-inference-server) to host it elsewhere, otherwise the training script will instantiate it ([a default one](https://huggingface.co/Dahoas/gptj-rm-static)) on its own.

Launch training of [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) on 7 GPUs with 8th GPU hosting a reward model:
```sh
accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py
```
Or if you want to train a smaller model or start from a supervised checkpoint, you can use one of the [configs](../../configs)
```sh
CONFIG_NAME=125M accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py
```

Already trained models are hosted on https://huggingface.co/reciprocate

#### Optional steps to setup a reward model (trained with [Dahoas/reward-modeling](https://github.com/Dahoas/reward-modeling)) with Triton Server:

```sh
# convert the model and create a config and a folder `model_store` structured for Triton
python to_triton.py --base_model EleutherAI/gpt-j-6B --checkpoint Dahoas/gptj-rm-static --revision 676bfd4d

# convert the docker image (skip this if you use docker instead)
singularity build --sandbox tritonserver-pyt.sif docker://nvcr.io/nvidia/tritonserver:22.08-pyt-python-py3
```

```sh
# start Triton Server pointing to the `model_store` containing the reward model
SINGULARITYENV_CUDA_VISIBLE_DEVICES=7 singularity run --nv --bind model_store:/model_store tritonserver-pyt.sif tritonserver --model-repository=/model_store &

# set model's url and replace the name after the slash if you use a different checkpoint
export TRITON_HOST=localhost:8001/gptj-rm-static

# launch training
accelerate launch --num_processes 7 --config_file ../../configs/accelerate/zero2-bf16.yaml ppo_hh.py
```

#### Sample W&B runs

PPO GPT-J: https://wandb.ai/sorry/trlx/runs/v0bir5s9

ILQL GPT-J: https://wandb.ai/sorry/trlx/runs/1qqxp72a

SFT GPT-J: https://wandb.ai/sorry/trlx/runs/a7ng078v

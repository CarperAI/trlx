### Training on Anthropic's Helpful & Harmless [dataset](https://github.com/anthropics/hh-rlhf)

As an example, the following setup assumes a single machine with 8xA100 80GB, the last of which will be dedicated to hosting a reward model. Optionally you can use [Triton Inference Server](https://github.com/triton-inference-server) to host it elsewhere, otherwise the training script will instantiate it (a default one) on its own. Steps to setup a reward model (trained with [Dahoas/reward-modeling](https://github.com/Dahoas/reward-modeling)) with Triton Server:

```sh
# convert the model and create a config and a folder `model_store` structured for triton
python to_triton.py --base_model EleutherAI/gpt-j-6B --checkpoint Dahoas/gptj-rm-static --revision 676bfd4d

# convert the docker image (skip this if you use docker instead)
singularity build --sandbox tritonserver-pyt.sif docker://nvcr.io/nvidia/tritonserver:22.08-pyt-python-py3
```

```sh
# start triton server pointing to the folder with a model
SINGULARITYENV_CUDA_VISIBLE_DEVICES=7 singularity run --nv --bind model_store:/model_store tritonserver-pyt.sif tritonserver --model-repository=/model_store &

# set model's url and replace the name after the slash if you use a different checkpoint
export TRITON_HOST=localhost:8001/gptj-rm-static
```

Launch training on 7 GPUs with 8th GPU hosting a reward model
```sh
accelerate launch --num_processes 7 --config_file ../../configs/deepspeed/zero2-bf16.yaml ppo_hh.py
```

## StableVicuna: Open Source RLHF LLM Chatbot

### Dataset

1. Reward Dataset:
- Download at [this](https://huggingface.co/datasets/reciprocate/oasst_hh_shp_hellaswag_webgpt_rm_dataset)
- Dataset size:
    - Train: 264534 samples
    - Valid: 2874 samples

2.  SFT and RL Prompt Dataset:
- Download at [this](https://huggingface.co/datasets/pvduy/stable_vicuna_oasst_format)
- Dataset size: 89991 samples

### Reward Model Training
To train reward model you can following instruction from this [repo](https://github.com/CarperAI/autocrit)

Command:
```bash
python preference.py --model_path reciprocate/dahoas-gptj-rm-static --dataset reciprocate/oasst_hh_shp_hellaswag_webgpt_rm_dataset --batch_size 2 --eval_interval 500 --lr 0.000001
```

### RL Training:

1. Distributed Training:

We trained on 4 nodes with 8 A100 GPUs each.


```bash
sbatch go_train_dist.sh
```

WANDB runs: https://wandb.ai/pvduy/trlx/runs/w8d20kam

2. Single Node Training:
In case that you want to train on a single node, you can use the following command, but we do not garantee the result.

```bash
accelerate launch examples/rl_training.py
```

Accelerate config:
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: no
dynamo_config: {}
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
mixed_precision: bf16
num_machines: 1
num_processes: 7
rdzv_backend: static
same_network: true
use_cpu: false
```

### Released Model and Result:
You can find more details [here](https://huggingface.co/pvduy/stable-vicuna-13b-version2)

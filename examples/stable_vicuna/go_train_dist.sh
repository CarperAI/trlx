#!/bin/bash
#SBATCH --job-name=dist_vic
#SBATCH --partition=a100-cu117
#SBATCH --account=stablegpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=out_dist_vic_test_5.txt
#SBATCH --exclusive

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
# export CUDA_LAUNCH_BLOCKING=1

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=13043
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM=false
# export TRITON_HOST=localhost:8001

srun train_dist.sh

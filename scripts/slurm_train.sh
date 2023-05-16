#!/bin/bash
#SBATCH --job-name=trlx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=g40
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment=carperai
#SBATCH --exclusive

# Example usage:
# sbatch slurm_train.sh TRLX_DIR

set -exuo pipefail

export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CRE DITS=64
# export CUDA_LAUNCH_BLOCKING=1

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1234
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

TRLX_DIR=${1:-/fsx/home-amuzio/trlx}
TRAIN_SCRIPT=${2-scripts/accelerate_train_example.sh} # relative to TRLX_DIR
CONFIG_FILE=${3-configs/accelerate/zero2-bf16.yaml} # relative to TRLX_DIR
CONDA_DIR=${4:-/admin/home-amuzio/miniconda3}
CONDA_ENV_NAME=${5:-trlx}

pushd $TRLX_DIR
srun --comment carperai $TRAIN_SCRIPT \
        $CONFIG_FILE \
        $CONDA_DIR \
        $CONDA_ENV_NAME

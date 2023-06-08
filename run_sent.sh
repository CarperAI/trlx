#!/bin/bash
#SBATCH --job-name=nemo-trlx
#SBATCH --partition=a100-cu117
#SBATCH --account=stablegpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --output=nemo_out.txt
#SBATCH --error=error_nemo.txt
#SBATCH --exclusive
#SBATCH --exclude=cw-prod-a100-cu117-34

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
ulimit -c unlimited


export NCCL_DEBUG=INFO

export NCCL_COLLNET_ENABLE=0


source /mnt/nvme/home/duyphung/.bashrc

conda env list
eval "$(conda shell.bash hook)"
conda activate nemo_convert 

cd /mnt/hdd/duyphung/nemo_converter/trlx
export WANDB_ENTITY=pvduy  
srun python examples/nemo_ppo_sentiments.py

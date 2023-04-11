#!/bin/bash
#SBATCH --job-name=trlx-sweep
#SBATCH --account=trlx
#SBATCH --partition=a100-cu117
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=%j
#SBATCH --exclusive

export NCCL_DEBUG=WARN
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
# export CUDA_LAUNCH_BLOCKING=1

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

cd $TRLX
source $TRLX/venv-with-pinned-ray/bin/activate

ray start --head --port=6379 &

export HOSTNAMES=($HOSTNAMES)
for node in ${HOSTNAMES[@]:1}; do
    echo "Starting ray worker @ $node"
    srun --nodes=1 --ntasks=1 -w "$node" ray start --address $MASTER_ADDR:6379 --block &
done

sleep 10
ray status

NUM_GPUS=16
python -m trlx.sweep -y --config configs/sweeps/ppo_sweep.yml --accelerate_config configs/accelerate/zero2-bf16.yaml --num_gpus $NUM_GPUS examples/ppo_sentiments.py
# python -m trlx.sweep -y --config configs/sweeps/ilql_sweep.yml --default_config configs/ilql_config.yml --accelerate_config configs/accelerate/zero2-bf16.yaml --num_gpus $NUM_GPUS examples/ilql_sentiments.py

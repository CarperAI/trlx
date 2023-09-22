#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --cpus-per-task=8
#SBATCH --output=out.txt
#SBATCH --error=error.txt
#SBATCH --exclusive

cd examples/llama_nemo
srun --label python nemo_llama2_ppo_sentiments.py

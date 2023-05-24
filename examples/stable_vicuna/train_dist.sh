#!/bin/bash

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo hostname = `hostname`
echo HOSTNAMES = $HOSTNAMES
echo MASTER_ADDR = $MASTER_ADDR
echo MASTER_PORT = $MASTER_PORT
echo RANK = $RANK
echo COUNT_NODE = $COUNT_NODE


conda env list
eval "$(conda shell.bash hook)"
conda activate trlx_env

cd trlx/examples/stable_vicuna

if [[ $RANK -eq 0 ]]; then
    accelerate launch --num_processes $((8 * $COUNT_NODE - 1)) --num_machines $COUNT_NODE --machine_rank $RANK --main_process_port 1234 --main_process_ip $MASTER_ADDR --config_file configs/accelerate/zero2-bf16.yaml rl_training.py
else
    accelerate launch --num_processes $((8 * $COUNT_NODE)) --num_machines $COUNT_NODE --machine_rank $RANK --main_process_port 1234 --main_process_ip $MASTER_ADDR --config_file configs/accelerate/zero2-bf16.yaml rl_training.py
fi

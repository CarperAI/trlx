#!/bin/bash

set -exuo pipefail

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script
H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

CONFIG_FILE=${1-configs/deepspeed/zero2-bf16.yaml} # relative to TRLX_DIR
CONDA_DIR=${2:-/admin/home-amuzio/miniconda3}
CONDA_ENV_NAME=${3:-trlx}

# This script assumes the following:
# (1) a conda environment named $CONDA_ENV_NAME
# (2) It is being run from the $TRLX_DIR directory
# If using venv, you can remove the conda stuff and just activate the venv directly
set +x
export PATH="$CONDA_DIR/condabin:$PATH"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
set -x


accelerate launch \
    --num_processes $((8 * $COUNT_NODE)) \
    --num_machines $COUNT_NODE \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --config_file $CONFIG_FILE \
    examples/ilql_sentiments.py

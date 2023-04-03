#!/bin/bash
set -e

origin=CarperAI/trlx
branch=main
entity=null
only_hash=false
only_tiny=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --origin) origin="$2"; shift ;;
        --branch) branch="$2"; shift ;;
        --public) entity='"CarperAI"' ;;
        --only_hash) only_hash=true ;;
        --only_tiny) only_tiny=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

dir=`mktemp -d -p .`
if [ ! -d "$dir" ]; then
   echo "Couldn't create a temporary directory, aborting"
   exit 1
fi

cd $dir
trap "rm -rf ../$dir" EXIT

git clone --depth 1 --single-branch -b $branch https://github.com/$origin .

hash=`find . -not \( -path ./.git -prune \) -not -name "*.md" -type f -print0 | sort -z | xargs -0 sha1sum | sha1sum | cut -f1 -d" "`
git_hash=`git log --format=%h/%s/%as -n1`

if [ "$only_hash" = true ]; then
   echo "$hash"
   echo "$git_hash"
   exit 0
fi

python -m venv venv
. venv/bin/activate
python -m pip install pip --upgrade
pip install -r requirements.txt
pip install -e .

args='{"train": {"project_name": "trlx-references", "entity_name": '$entity', "tags": ["'$hash'"]}}'
python examples/randomwalks/ilql_randomwalks.py "$args"
python examples/randomwalks/ppo_randomwalks.py "$args"

if [ "$only_tiny" = true ]; then
    exit 0
fi

rm -rf ../benchmark_logs && mkdir ../benchmark_logs

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --config_file configs/accelerate/zero2-bf16.yaml --main_process_port 8880 examples/ppo_sentiments.py "$args" > ../benchmark_logs/ppo_sentiments.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --config_file configs/accelerate/zero2-bf16.yaml --main_process_port 8881 examples/sft_sentiments.py "$args" > ../benchmark_logs/sft_sentiments.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --config_file configs/accelerate/zero2-bf16.yaml --main_process_port 8882 examples/ilql_sentiments.py "$args" > ../benchmark_logs/ilql_sentiments.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 --config_file configs/accelerate/zero2-bf16.yaml --main_process_port 8883 examples/ppo_sentiments_t5.py "$args" > ../benchmark_logs/ppo_sentiments_t5.log 2>&1 &

wait

args='{"train": {"total_steps": 1500, "seq_length": 512, "project_name": "trlx-references", "entity_name": '$entity', "tags": ["'$hash'"]}}'
CONFIG_NAME=6B accelerate launch --num_processes 7 --config_file configs/accelerate/zero2-bf16.yaml examples/hh/ppo_hh.py "$args"

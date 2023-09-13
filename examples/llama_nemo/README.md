### NeMo Megatron setup:

- Install NeMo version: v1.17.0

```bash
git clone https://github.com/NVIDIA/NeMo/
cd NeMo
git checkout d3017e4
pip install -e '.[all]'
```

- Install Apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### Convert LLaMa to NeMo:
Example:

```bash
python convert_llama_to_nemo.py --model_path NousResearch/Llama-2-7b-hf --output_folder nemo_llama2_7b --total_tp 4  --name 7b
```

### Training:
Example: [wandb](https://wandb.ai/carperai/trlxnemo/runs/v7592y73?workspace=user-pvduy)

```bash
sbatch dist_train.sh
```

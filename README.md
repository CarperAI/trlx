# Soft Optimization

## Use

### Setup

First run the setup script (replace `-j` with the correct user):

```bash
sh ./setup.sh -j
```

### Running a script

To run directly:

```bash
poetry run python soft_optim/fine_tune.py
```

To launch Accelerate use:

```bash
accelerate launch --config_file configs/deepspeed_configs/default_configs.yml examples/simulacra_tmp.py
```
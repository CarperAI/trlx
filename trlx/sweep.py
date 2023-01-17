# python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml examples/ppo_sentiments.py
import argparse
import os
import importlib
from pathlib import Path

import ray
import yaml
from ray import tune
from ray.air import ScalingConfig
from ray.tune.logger import CSVLoggerCallback

from trlx.ray_tune import get_param_space, get_tune_config
from trlx.utils import get_git_tag

# from trlx.ray_tune.wandb import create_report, log_trials

from trlx.ray_train.accelerate_trainer import AccelerateTrainer


def tune_function(
    train_function,
    param_space: dict,
    tune_config: dict,
    default_config: dict,
    resources: dict,
):
    num_workers = resources.pop("num_workers")
    param_space["default_config"] = default_config.copy()
    param_space["default_config"]["train"]["git_tag"] = get_git_tag()
    param_space_train = {"train_loop_config": param_space}

    tuner = tune.Tuner(
        AccelerateTrainer(
            train_function,
            accelerate_config_path=None,  # Mandatory arg. None means use Accelerate default path
            scaling_config=ScalingConfig(
                trainer_resources={"CPU": 0},
                num_workers=num_workers,
                use_gpu=bool(resources["GPU"]),
                resources_per_worker=resources,
            ),
        ),
        param_space=param_space_train,
        tune_config=tune.TuneConfig(**tune_config),
        run_config=ray.air.RunConfig(
            local_dir="ray_results", callbacks=[CSVLoggerCallback()]
        ),
    )
    results = tuner.fit()

    project_name = tune_config.get("project_name", "sweep")

    # log_trials(
    #     tuner._local_tuner.get_experiment_checkpoint_dir(),
    #     project_name,
    # )

    # create_report(
    #     project_name,
    #     param_space,
    #     tune_config,
    #     Path(tuner._local_tuner.get_experiment_checkpoint_dir()).stem,
    #     results.get_best_result().config,
    # )

    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str, help="Path to the script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file defining the param_space.",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        required=True,
        help="The default config file for the script.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of workers to use per trial."
    )
    parser.add_argument(
        "--num-cpus", type=int, default=4, help="Number of CPUs to use per worker."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use per worker."
    )
    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Don't ask for confirmation"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )

    args, _ = parser.parse_known_args()

    # Read config and parse it
    with open(args.config) as f:
        config = yaml.safe_load(f)
    tune_config = get_tune_config(config.pop("tune_config"))
    param_space = get_param_space(config)
    with open(args.default_config) as f:
        default_config = yaml.safe_load(f)

    resources = {
        "num_workers": args.num_workers,
        "CPU": args.num_cpus,
        "GPU": args.num_gpus,
    }

    print(f'WARNING: Importing main from "{args.script}" and everything along with it')

    if not args.assume_yes:
        print("Please confirm y/n: ", end="")
        if input() != "y":
            print("Exiting")
            exit(1)

    # convert a nested path to a module path
    script_path = args.script.replace(".py", "").replace("/", ".")
    script = importlib.import_module(script_path)
    # Register the training function that will be used for training the model.
    # tune.register_trainable("train_function", script.main)
    tune_function(script.main, param_space, tune_config, default_config, resources)

    # Shut down Ray.
    ray.shutdown()

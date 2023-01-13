# python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml examples/ppo_sentiments.py
import argparse
import os
import importlib
from pathlib import Path

import ray
import yaml
from ray import tune
from ray.air import ScalingConfig, session
from ray.train.torch import TorchTrainer
from ray.tune.logger import CSVLoggerCallback
import tempfile

from trlx.ray_tune import get_param_space, get_tune_config
from accelerate.commands.config import default_config_file, load_config_from_file
from trlx.ray_train.launch import launch_command, launch_command_parser

# from trlx.ray_tune.wandb import create_report, log_trials

from argparse import Namespace
class DefaultNamespace(Namespace):
    def __getattr__(self, name: str):
        parser = launch_command_parser()
        ret = parser.get_default(name)
        return ret

def tune_function(
    train_function, param_space: dict, tune_config: dict, resources: dict
):
    default_config = yaml.safe_load(open("configs/ppo_config.yml"))
    param_space["default_config"] = default_config

    config_file_path = default_config_file
    with open(config_file_path, "r") as f:
        config_data = f.read()

    def train_function_wrapper(config):
        temp_config_file = tempfile.mkstemp()[1]
        with open(temp_config_file, "w") as f:
            f.write(config_data)
        args = DefaultNamespace()
        setattr(args, "config_file", temp_config_file)
        launch_command(args)
        os.environ["RANK"] = str(session.get_world_rank())
        os.environ["WORLD_RANK"] = str(session.get_world_rank())
        os.environ["LOCAL_RANK"] = str(session.get_local_rank())
        os.environ["WORLD_SIZE"] = str(session.get_world_size())
        os.environ["LOCAL_WORLD_SIZE"] = str(session.get_local_world_size())
        os.environ["CROSS_RANK"] = str(session.get_world_rank())
        os.environ["CROSS_SIZE"] = str(session.get_world_size())
        os.environ["LOCAL_SIZE"] = str(session.get_local_world_size())
        print(os.environ)

        return train_function(config)

    param_space_train = {"train_loop_config": param_space}
    tuner = tune.Tuner(
        TorchTrainer(
            train_function_wrapper,
            scaling_config=ScalingConfig(
                trainer_resources={"CPU": 0},
                num_workers=resources["gpu"],
                use_gpu=bool(resources["gpu"]),
                resources_per_worker={"CPU": resources["cpu"], "GPU": int(bool(resources["gpu"]))},
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
        "--num-cpus", type=int, default=4, help="Number of CPUs to use per exp."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use per exp."
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

    resources = {
        "cpu": args.num_cpus,
        "gpu": args.num_gpus,
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
    tune_function(script.main, param_space, tune_config, resources)

    # Shut down Ray.
    ray.shutdown()

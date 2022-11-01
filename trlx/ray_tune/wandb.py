"""Utility function to log the results of a Ray Tune experiment to W&B."""

import os
import wandb
import json
import pandas as pd


def log_trials(trial_path: str, project_name: str):
    files = os.listdir(trial_path)

    trial_paths = []
    for filename in files:
        tmp_path = os.path.join(trial_path, filename)
        if os.path.isdir(tmp_path):
            trial_paths.append(tmp_path)

    for trial in trial_paths:
        files = os.listdir(trial)
        # Open params.json and load the configs for that trial.
        with open(os.path.join(trial, "params.json"), "r") as f:
            params = json.load(f)
        # Initialize wandb
        run = wandb.init(
            project=project_name,
            config=dict(params),
            group=trial_path,
            job_type="hyperopt",
        )
        # Open result.json and log the metrics to W&B.
        with open(os.path.join(trial, "result.json"), "r") as f:
            for line in f:
                result = dict(json.loads(line))
                result.pop("config", None)
                result = parse_result(result)
                wandb.log(result)

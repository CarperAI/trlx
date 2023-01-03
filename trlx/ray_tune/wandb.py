"""Utility function to log the results of a Ray Tune experiment to W&B."""

import json
import math
import os
from pathlib import Path

import wandb

wandb.require("report-editing")
import wandb.apis.reports as wb  # noqa: E402

ray_info = [
    "done",
    "time_this_iter_s",
    "timesteps_total",
    "episodes_total",
    "iterations_since_restore",
    "timesteps_since_restore",
    "time_since_restore",
    "warmup_time",
    "should_checkpoint",
    "training_iteration",
    "timestamp",
    "pid",
]


def parse_result(result):
    out = {}
    for k, v in result.items():
        if (
            isinstance(v, (int, float))
            and not k.startswith("config.")
            and k not in ray_info
        ):
            out[k] = v

    return out


def significant(x):
    return round(x, 1 - int(math.floor(math.log10(x))))


def log_trials(trial_path: str, project_name: str):
    trial_path = Path(trial_path)
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

        name = ",".join(f"{k}={significant(v)}" for k, v in params.items())
        # Initialize wandb
        run = wandb.init(
            name=name,
            project=project_name,
            config=params,
            group=trial_path.stem,
            job_type="hyperopt",
        )

        # Open result.json and log the metrics to W&B.
        with open(os.path.join(trial, "result.json"), "r") as f:
            for line in f:
                result = json.loads(line)
                result.pop("config", None)
                wandb.log(parse_result(result))

        # Close the W&B run.
        run.finish()


def create_report(project_name, param_space, tune_config, trial_path, best_config=None):
    def get_parallel_coordinate(param_space, metric):
        column_names = list(param_space.keys())
        columns = [wb.reports.PCColumn(column) for column in column_names]

        return wb.ParallelCoordinatesPlot(
            columns=columns + [wb.reports.PCColumn(metric)],
            layout={"x": 0, "y": 0, "w": 12 * 2, "h": 5 * 2},
        )

    def get_param_importance(metric):
        return wb.ParameterImportancePlot(
            # Get it from the metric name.
            with_respect_to=metric,
            layout={"x": 0, "y": 5, "w": 6 * 2, "h": 4 * 2},
        )

    def get_scatter_plot(metric):
        return wb.ScatterPlot(
            # Get it from the metric name.
            title=f"{metric} v. Index",
            x="Index",
            y=metric,
            running_ymin=True,
            font_size="small",
            layout={"x": 6, "y": 5, "w": 6 * 2, "h": 4 * 2},
        )

    def get_metrics_with_history(project_name, group_name, entity=None):
        entity_project = f"{entity}/{project_name}" if entity else project_name
        api = wandb.Api()
        runs = api.runs(entity_project)

        runs = sorted(
            runs,
            key=lambda run: run.summary.get(tune_config["metric"], -math.inf),
            reverse=True,
        )

        for run in runs:
            if run.group == str(group_name):
                history = run.history()
                metrics = history.columns
                break

        metrics = [metric for metric in metrics if not metric.startswith("_")]
        return metrics

    report = wb.Report(
        project=project_name,
        title=f"Hyperparameter Optimization Report: {trial_path}",
        description="This is a report that shows the results of a hyperparameter optimization experiment.",
    )

    report.blocks = [
        wb.P(
            "The following plots show the results of the hyperparameter optimization experiment. "
            "Use this as a starting point for your analysis. Go in the edit mode to customize the report. "
            "Share it with your team to collaborate on the analysis."
        ),
        wb.H1(text="Analysis"),
        wb.P(
            "Parallel coordinates chart (top) summarize the relationship between large numbers of hyperparameters "
            "and model metrics at a glance. \nThe scatter plot (right) compares the different trials and gives you a "
            "insight on how the trials progressed. \nThe parameter importance plot(left) lists the hyperparameters "
            "that were the best predictors of, and highly correlated to desirable values of your metrics."
        ),
        wb.PanelGrid(
            panels=[
                get_parallel_coordinate(param_space, tune_config["metric"]),
                get_param_importance(tune_config["metric"]),
                get_scatter_plot(tune_config["metric"]),
            ],
            runsets=[
                wb.RunSet(project=project_name).set_filters_with_python_expr(
                    f'group == "{trial_path}"'
                )
            ],
        ),
    ]

    metrics = get_metrics_with_history(
        project_name,
        trial_path,
    )

    line_plot_panels = []
    for metric in metrics:
        line_plot_panels.append(
            wb.LinePlot(
                title=f"{metric}",
                x="Step",
                y=[f"{metric}"],
                title_x="Step",
                smoothing_show_original=True,
                max_runs_to_show=10,
                plot_type="line",
                font_size="auto",
                legend_position="north",
            )
        )

    report.blocks = report.blocks + [
        wb.H1(text="Metrics"),
        wb.P(
            "The following line plots show the metrics for each trial. Use this to investigate the "
            "performance of the model for each trial at the metrics level."
        ),
        wb.PanelGrid(
            panels=line_plot_panels,
            runsets=[
                wb.RunSet(project=project_name).set_filters_with_python_expr(
                    f'group == "{trial_path}"'
                )
            ],
        ),
    ]

    if best_config:
        report.blocks = report.blocks + [
            wb.H1(text="Best Config"),
            wb.P(
                "The code block shown below is the best config found by the hyperparameter "
                "optimization experiment according to Ray Tune."
            ),
            wb.CodeBlock(code=[json.dumps(best_config, indent=4)], language="json"),
        ]

    report.save()
    print(report.url)

# python scripts/reference.py --pr_branch CarperAI/trlx:convert-examples-configs --ref_branch CarperAI/trlx:main

import argparse
import os
import subprocess
import time

import wandb
import wandb.apis.reports as wb

parser = argparse.ArgumentParser()
parser.add_argument("--pr_branch", type=str, required=True, help="Git branch of the format (origin:branch)")
parser.add_argument("--ref_branch", type=str, default="CarperAI/trlx:main", help="Reference git branch")
parser.add_argument("--public", action="store_true", help="Use CarperAI entity to store w&b runs")
args = parser.parse_args()

pr_origin = ref_origin = "CarperAI/trlx"
pr_branch = args.pr_branch
ref_branch = args.ref_branch
if ':' in pr_branch:
    pr_origin, pr_branch = pr_branch.rsplit(':', 1)
if ':' in ref_branch:
    ref_origin, ref_branch = ref_branch.rsplit(':', 1)

pr_hash = os.popen(f"./scripts/benchmark.sh --origin {pr_origin} --branch {pr_branch} --only_hash").read()[:-1]
ref_hash = os.popen(f"./scripts/benchmark.sh --origin {ref_origin} --branch {ref_branch} --only_hash").read()[:-1]

api = wandb.Api()
project_name = "CarperAI/trlx-references" if args.public else "trlx-references"
public = "--public" if args.public else ""

runs = api.runs(project_name, filters={"tags": {"$in": [ref_hash]}})
if runs:
    print(f"On {ref_branch} these runs were already made: \n{chr(10).join(run.name for run in runs)}")
else:
    print(f"Making runs on {ref_branch}")
    subprocess.run(f"./scripts/benchmark.sh --origin {ref_origin} --branch {ref_branch} {public}".split())

runs = api.runs(project_name, filters={"tags": {"$in": [pr_hash]}})
if runs:
    print(f"On {pr_branch} these runs were already made: \n{chr(10).join(run.name for run in runs)}")
else:
    print(f"Making runs on {pr_branch}")
    subprocess.run(f"./scripts/benchmark.sh --origin {pr_origin} --branch {pr_branch} {public}".split())

# wait for a bit until w&b syncs runs
time.sleep(10)


report = wb.Report(
    # entity="carperai" if args.public else None,
    project=project_name.split('/')[1] if args.public else project_name,
    title=f"{pr_branch} v. {ref_branch}",
)

# collect metric columns to display
metrics = set(sum([[metric for metric in run.history().columns if not metric.startswith("_")] for run in runs], []))
metrics_panels = [
    wb.LinePlot(
        title=f"{metric}",
        x="Step",
        y=[f"{metric}"],
        title_x="Step",
        smoothing_show_original=True,
        max_runs_to_show=100,
        plot_type="line",
        font_size="auto",
        legend_position="north",
    ) for metric in metrics
]

# sort the most important metrics to be shown first
major_metrics = set()
for metric in metrics:
    if metric.startswith("reward") or metric.startswith("metric"):
        major_metrics.add(metric)
metrics = metrics - major_metrics

report.blocks = [
    wb.H1(text="Metrics"),
    wb.PanelGrid(
        panels=[panel for panel in metrics_panels if panel.title in major_metrics],
        runsets=[wb.Runset(
            project=project_name,
            filters={"tags": {"$in": [pr_hash, ref_hash]}}
        )],
    ),
    wb.PanelGrid(
        panels=[panel for panel in metrics_panels if panel.title in metrics],
        runsets=[wb.Runset(
            project=project_name,
            filters={"tags": {"$in": [pr_hash, ref_hash]}}
        )],
    ),
]

report.save()
print(report.url)

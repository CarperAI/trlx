# python -m trlx.reference CarperAI/trlx:add-benchmark-tools --against CarperAI/trlx:main

import argparse
import os
import subprocess

import wandb
import wandb.apis.reports as wb

parser = argparse.ArgumentParser()
parser.add_argument("branch", type=str, help="Git branch in the format `origin:branch`")
parser.add_argument("--against", type=str, default="CarperAI/trlx:main", help="Reference git branch")
parser.add_argument("--public", action="store_true", help="Use CarperAI entity to store/pull from w&b runs")
args = parser.parse_args()

pr_origin = ref_origin = "CarperAI/trlx"
pr_branch = args.branch
ref_branch = args.against
if ":" in pr_branch:
    pr_origin, pr_branch = pr_branch.rsplit(":", 1)
if ":" in ref_branch:
    ref_origin, ref_branch = ref_branch.rsplit(":", 1)

out = os.popen(f"./scripts/benchmark.sh --origin {pr_origin} --branch {pr_branch} --only_hash")
pr_hash, pr_git_hash = [x[:-1] for x in out.readlines()]

out = os.popen(f"./scripts/benchmark.sh --origin {ref_origin} --branch {ref_branch} --only_hash")
ref_hash, ref_git_hash = [x[:-1] for x in out.readlines()]

print(f"{pr_origin}:{pr_branch=} {pr_hash=} {pr_git_hash=}")
print(f"{ref_origin}:{ref_branch} {ref_hash=} {ref_git_hash=}")

api = wandb.Api()
project_name = "CarperAI/trlx-references" if args.public else "trlx-references"
public = "--public" if args.public else ""

runs = api.runs(project_name, filters={"tags": {"$in": [ref_hash]}})
if runs:
    print(f"On {ref_branch} @{ref_git_hash} these runs were already made: \n{chr(10).join(run.name for run in runs)}")
else:
    print(f"Making runs on {ref_branch} @{ref_git_hash}")
    subprocess.run(f"./scripts/benchmark.sh --origin {ref_origin} --branch {ref_branch} {public}".split())

runs = api.runs(project_name, filters={"tags": {"$in": [pr_hash]}})
if runs:
    print(f"On {pr_branch} @{pr_git_hash} these runs were already made: \n{chr(10).join(run.name for run in runs)}")
else:
    print(f"Making runs on {pr_branch} @{pr_git_hash}")
    subprocess.run(f"./scripts/benchmark.sh --origin {pr_origin} --branch {pr_branch} {public}".split())

report = wb.Report(
    project=project_name.split("/")[1] if args.public else project_name,
    title=f"{pr_branch} v. {ref_branch}",
    description=f"{pr_branch}\n@{pr_git_hash}\n\n{ref_branch}\n@{ref_git_hash}",
)
blocks = []

experiment_names = set(x.name.split(":")[0] for x in api.runs(project_name))
for name in experiment_names:
    filters = {"$and": [{"display_name": {"$regex": f"^{name}"}}, {"tags": {"$in": [pr_hash, ref_hash]}}]}

    runs = api.runs(project_name, filters=filters)
    metrics = set(sum([[metric for metric in run.history().columns if not metric.startswith("_")] for run in runs], []))

    metrics_panels = [
        wb.LinePlot(
            title=f"{metric}",
            x="Step",
            y=[metric],
            title_x="Step",
            smoothing_show_original=True,
            max_runs_to_show=2,
            plot_type="line",
            font_size="auto",
            legend_position="north",
        )
        for metric in metrics
    ]

    # sort the most important metrics to be shown first
    major_metrics = set()
    for metric in metrics:
        if metric.startswith("reward") or metric.startswith("metric"):
            major_metrics.add(metric)
    metrics = metrics - major_metrics

    blocks.extend(
        [
            wb.H1(text=name),
            wb.PanelGrid(
                panels=[panel for panel in metrics_panels if panel.title in major_metrics],
                runsets=[wb.Runset(project=project_name, filters=filters)],
            ),
            wb.PanelGrid(
                panels=[panel for panel in metrics_panels if panel.title in metrics],
                runsets=[wb.Runset(project=project_name, filters=filters)],
            ),
        ]
    )

report.blocks = blocks
report.save()
print(report.url)

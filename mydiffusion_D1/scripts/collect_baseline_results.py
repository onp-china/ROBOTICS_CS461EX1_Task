#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys

MYDIFFUSION_D1_ROOT = Path(__file__).resolve().parents[1]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import (
    MIMICGEN_TASKS,
    add_repo_paths,
    mean,
    outputs_dir,
    population_std,
    read_json_lines,
    reports_dir,
    task_config_name,
)


CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>\d+)-(?P<metric>[A-Za-z0-9_]+)=(?P<value>-?\d+(?:\.\d+)?)\.ckpt$")
CHECKPOINT_METRIC_MODES = {
    "test_mean_score": "max",
    "val_loss": "min",
    "test_mean_min_eef_object_distance": "min",
}
LEGACY_EXP_NAME = "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MimicGen D1 baseline results into CSV and Markdown reports.")
    parser.add_argument("--output-dir", default=str(outputs_dir()))
    return parser.parse_args()


def parse_checkpoint_name(checkpoint_name: str) -> tuple[int, str, float] | None:
    match = CHECKPOINT_RE.match(checkpoint_name)
    if not match:
        return None
    metric_name = str(match.group("metric"))
    if metric_name not in CHECKPOINT_METRIC_MODES:
        return None
    return int(match.group("epoch")), metric_name, float(match.group("value"))


def format_checkpoint_metric(metric_name: str | None, value: float | None) -> str:
    if metric_name is None or value is None:
        return "-"
    if metric_name == "test_mean_score":
        return f"test/mean_score={value:.6f}"
    if metric_name == "val_loss":
        return f"val_loss={value:.6f}"
    if metric_name == "test_mean_min_eef_object_distance":
        return f"test/mean_min_eef_object_distance={value:.6f}"
    return f"{metric_name}={value:.6f}"


def _improves(metric_name: str, candidate_value: float, best_value: float | None, candidate_epoch: int, best_epoch: int) -> bool:
    if best_value is None:
        return True
    mode = CHECKPOINT_METRIC_MODES[metric_name]
    if mode == "max":
        return candidate_value > best_value or (candidate_value == best_value and candidate_epoch > best_epoch)
    return candidate_value < best_value or (candidate_value == best_value and candidate_epoch > best_epoch)


def summarize_run(run_dir: Path, task_name: str, exp_name: str, seed: int) -> dict | None:
    log_path = run_dir / "logs.json.txt"
    if not log_path.is_file():
        return None

    rows = read_json_lines(log_path)
    if not rows:
        return None

    best_test_mean_score = None
    final_test_mean_score = None
    final_val_loss = None
    best_test_contact_rate = None
    best_test_mean_min_eef_object_distance = None
    for row in rows:
        if "test/mean_score" in row:
            score = float(row["test/mean_score"])
            final_test_mean_score = score
            best_test_mean_score = score if best_test_mean_score is None else max(best_test_mean_score, score)
        if "val_loss" in row:
            final_val_loss = float(row["val_loss"])
        if "test/contact_rate" in row:
            contact_rate = float(row["test/contact_rate"])
            best_test_contact_rate = (
                contact_rate if best_test_contact_rate is None else max(best_test_contact_rate, contact_rate)
            )
        if "test/mean_min_eef_object_distance" in row:
            distance = float(row["test/mean_min_eef_object_distance"])
            if best_test_mean_min_eef_object_distance is None or distance < best_test_mean_min_eef_object_distance:
                best_test_mean_min_eef_object_distance = distance

    best_checkpoint = ""
    best_checkpoint_metric_name = None
    best_checkpoint_metric_value = None
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        best_epoch = -1
        for checkpoint in sorted(checkpoints_dir.glob("epoch=*-*.ckpt")):
            parsed = parse_checkpoint_name(checkpoint.name)
            if parsed is None:
                continue
            epoch, metric_name, metric_value = parsed
            if _improves(metric_name, metric_value, best_checkpoint_metric_value, epoch, best_epoch):
                best_checkpoint_metric_name = metric_name
                best_checkpoint_metric_value = metric_value
                best_epoch = epoch
                best_checkpoint = str(checkpoint)
        if not best_checkpoint and (checkpoints_dir / "latest.ckpt").is_file():
            best_checkpoint = str(checkpoints_dir / "latest.ckpt")

    return {
        "task": task_name,
        "exp_name": exp_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
        "best_checkpoint_metric_value": best_checkpoint_metric_value,
        "best_test_mean_score": best_test_mean_score,
        "final_test_mean_score": final_test_mean_score,
        "final_val_loss": final_val_loss,
        "best_test_contact_rate": best_test_contact_rate,
        "best_test_mean_min_eef_object_distance": best_test_mean_min_eef_object_distance,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "exp_name",
        "seed",
        "run_dir",
        "best_checkpoint",
        "best_checkpoint_metric_name",
        "best_checkpoint_metric_value",
        "best_test_mean_score",
        "final_test_mean_score",
        "final_val_loss",
        "best_test_contact_rate",
        "best_test_mean_min_eef_object_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MimicGen D1 Baseline Summary",
        "",
        "## Per-run",
        "",
        "| task | exp_name | seed | best checkpoint | best checkpoint metric | best `test/mean_score` | best `test/contact_rate` | best `test/mean_min_eef_object_distance` | final `test/mean_score` | final `val_loss` |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {exp_name} | {seed} | {best_checkpoint} | {best_checkpoint_metric} | {best_test_mean_score} | {best_test_contact_rate} | {best_test_mean_min_eef_object_distance} | {final_test_mean_score} | {final_val_loss} |".format(
                task=row["task"],
                exp_name=row["exp_name"],
                seed=row["seed"],
                best_checkpoint=row["best_checkpoint"] or "-",
                best_checkpoint_metric=format_checkpoint_metric(
                    row.get("best_checkpoint_metric_name"),
                    row.get("best_checkpoint_metric_value"),
                ),
                best_test_mean_score=format_optional(row["best_test_mean_score"]),
                best_test_contact_rate=format_optional(row["best_test_contact_rate"]),
                best_test_mean_min_eef_object_distance=format_optional(row["best_test_mean_min_eef_object_distance"]),
                final_test_mean_score=format_optional(row["final_test_mean_score"]),
                final_val_loss=format_optional(row["final_val_loss"]),
            )
        )

    aggregate_lines = []
    grouped_task_names = [task_config_name(task) for task in MIMICGEN_TASKS]
    for task_name in grouped_task_names:
        task_rows = [row for row in rows if row["task"] == task_name]
        exp_names = sorted({row["exp_name"] for row in task_rows})
        for exp_name in exp_names:
            exp_rows = [row for row in task_rows if row["exp_name"] == exp_name]
            seeds = sorted(row["seed"] for row in exp_rows)
            if seeds != [42, 43, 44]:
                continue
            scores = [row["best_test_mean_score"] for row in exp_rows if row["best_test_mean_score"] is not None]
            contacts = [row["best_test_contact_rate"] for row in exp_rows if row["best_test_contact_rate"] is not None]
            distances = [
                row["best_test_mean_min_eef_object_distance"]
                for row in exp_rows
                if row["best_test_mean_min_eef_object_distance"] is not None
            ]
            val_losses = [row["final_val_loss"] for row in exp_rows if row["final_val_loss"] is not None]
            aggregate_lines.append(
                "| {task} | {exp_name} | {score_mean} | {contact_mean} | {distance_mean} | {val_mean} |".format(
                    task=task_name,
                    exp_name=exp_name,
                    score_mean=f"{mean(scores):.6f}" if scores else "-",
                    contact_mean=f"{mean(contacts):.6f}" if contacts else "-",
                    distance_mean=f"{mean(distances):.6f}" if distances else "-",
                    val_mean=f"{mean(val_losses):.6f}" if len(val_losses) == 3 else "-",
                )
            )

    if aggregate_lines:
        lines.extend(
            [
                "",
                "## Three-seed aggregate",
                "",
                "| task | exp_name | mean best `test/mean_score` | mean best `test/contact_rate` | mean best `test/mean_min_eef_object_distance` | mean final `val_loss` |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
                *aggregate_lines,
            ]
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def discover_runs(output_root: Path, task_name: str) -> list[tuple[Path, str, int]]:
    task_dir = output_root / task_name
    if not task_dir.is_dir():
        return []

    runs: list[tuple[Path, str, int]] = []
    for child in sorted(task_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        try:
            seed = int(child.name)
        except ValueError:
            seed = None
        if seed is not None:
            runs.append((child, LEGACY_EXP_NAME, seed))
            continue
        exp_name = child.name
        for nested_dir in sorted(child.iterdir(), key=lambda path: path.name):
            if not nested_dir.is_dir():
                continue
            try:
                nested_seed = int(nested_dir.name)
            except ValueError:
                for variant_seed_dir in sorted(nested_dir.iterdir(), key=lambda path: path.name):
                    if not variant_seed_dir.is_dir():
                        continue
                    try:
                        variant_seed = int(variant_seed_dir.name)
                    except ValueError:
                        continue
                    runs.append((variant_seed_dir, exp_name, variant_seed))
                continue
            runs.append((nested_dir, exp_name, nested_seed))
    return runs


def main() -> None:
    add_repo_paths()
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()

    rows = []
    for task_name in MIMICGEN_TASKS:
        config_name = task_config_name(task_name)
        for run_dir, exp_name, seed in discover_runs(output_root, config_name):
            row = summarize_run(run_dir, config_name, exp_name, seed)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: (row["task"], row["exp_name"], row["seed"]))
    report_root = reports_dir()
    write_csv(rows, report_root / "baseline_summary.csv")
    write_markdown(rows, report_root / "baseline_summary.md")
    print(f"Wrote {report_root / 'baseline_summary.csv'}")
    print(f"Wrote {report_root / 'baseline_summary.md'}")
    print("Next: python mydiffusion_D1/scripts/plot_results.py")


if __name__ == "__main__":
    main()

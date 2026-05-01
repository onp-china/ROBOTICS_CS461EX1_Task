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


TEST_SCORE_CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>\d+)-test_mean_score=(?P<score>-?\d+(?:\.\d+)?)\.ckpt$")
VAL_LOSS_CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>\d+)-val_loss=(?P<loss>-?\d+(?:\.\d+)?)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MimicGen D1 baseline results into CSV and Markdown reports.")
    parser.add_argument("--output-dir", default=str(outputs_dir()))
    return parser.parse_args()


def summarize_run(run_dir: Path, task_name: str, seed: int) -> dict | None:
    log_path = run_dir / "logs.json.txt"
    if not log_path.is_file():
        return None

    rows = read_json_lines(log_path)
    if not rows:
        return None

    best_test_mean_score = None
    final_test_mean_score = None
    final_val_loss = None
    for row in rows:
        if "test/mean_score" in row:
            score = float(row["test/mean_score"])
            final_test_mean_score = score
            best_test_mean_score = score if best_test_mean_score is None else max(best_test_mean_score, score)
        if "val_loss" in row:
            final_val_loss = float(row["val_loss"])

    best_checkpoint = ""
    best_checkpoint_metric_name = None
    best_checkpoint_metric_value = None
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        best_epoch = -1
        for checkpoint in checkpoints_dir.glob("epoch=*-test_mean_score=*.ckpt"):
            match = TEST_SCORE_CHECKPOINT_RE.match(checkpoint.name)
            if not match:
                continue
            epoch = int(match.group("epoch"))
            score = float(match.group("score"))
            if (
                best_checkpoint_metric_value is None
                or score > best_checkpoint_metric_value
                or (score == best_checkpoint_metric_value and epoch > best_epoch)
            ):
                best_checkpoint_metric_name = "test_mean_score"
                best_checkpoint_metric_value = score
                best_epoch = epoch
                best_checkpoint = str(checkpoint)
        if not best_checkpoint:
            best_epoch = -1
            for checkpoint in checkpoints_dir.glob("epoch=*-val_loss=*.ckpt"):
                match = VAL_LOSS_CHECKPOINT_RE.match(checkpoint.name)
                if not match:
                    continue
                epoch = int(match.group("epoch"))
                val_loss = float(match.group("loss"))
                if (
                    best_checkpoint_metric_value is None
                    or val_loss < best_checkpoint_metric_value
                    or (val_loss == best_checkpoint_metric_value and epoch > best_epoch)
                ):
                    best_checkpoint_metric_name = "val_loss"
                    best_checkpoint_metric_value = val_loss
                    best_epoch = epoch
                    best_checkpoint = str(checkpoint)
        if not best_checkpoint and (checkpoints_dir / "latest.ckpt").is_file():
            best_checkpoint = str(checkpoints_dir / "latest.ckpt")

    return {
        "task": task_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
        "best_checkpoint_metric_value": best_checkpoint_metric_value,
        "best_test_mean_score": best_test_mean_score,
        "final_test_mean_score": final_test_mean_score,
        "final_val_loss": final_val_loss,
    }


def format_checkpoint_metric(metric_name: str | None, value: float | None) -> str:
    if metric_name is None or value is None:
        return "-"
    if metric_name == "test_mean_score":
        return f"test/mean_score={value:.6f}"
    if metric_name == "val_loss":
        return f"val_loss={value:.6f}"
    return f"{metric_name}={value:.6f}"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "seed",
        "run_dir",
        "best_checkpoint",
        "best_checkpoint_metric_name",
        "best_checkpoint_metric_value",
        "best_test_mean_score",
        "final_test_mean_score",
        "final_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MimicGen D1 Baseline Summary",
        "",
        "## Per-run",
        "",
        "| task | seed | best checkpoint | best checkpoint metric | best `test/mean_score` | final `test/mean_score` | final `val_loss` |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {seed} | {best_checkpoint} | {best_checkpoint_metric} | {best_test_mean_score} | {final_test_mean_score} | {final_val_loss} |".format(
                task=row["task"],
                seed=row["seed"],
                best_checkpoint=row["best_checkpoint"] or "-",
                best_checkpoint_metric=format_checkpoint_metric(
                    row.get("best_checkpoint_metric_name"),
                    row.get("best_checkpoint_metric_value"),
                ),
                best_test_mean_score=format_optional(row["best_test_mean_score"]),
                final_test_mean_score=format_optional(row["final_test_mean_score"]),
                final_val_loss=format_optional(row["final_val_loss"]),
            )
        )

    aggregate_lines = []
    for task_name in (task_config_name(task) for task in MIMICGEN_TASKS):
        task_rows = [row for row in rows if row["task"] == task_name]
        seeds = sorted(row["seed"] for row in task_rows)
        if seeds != [42, 43, 44]:
            continue
        scores = [row["best_test_mean_score"] for row in task_rows if row["best_test_mean_score"] is not None]
        if len(scores) != 3:
            scores = []
        val_losses = [row["final_val_loss"] for row in task_rows if row["final_val_loss"] is not None]
        aggregate_lines.append(
            "| {task} | {score_mean} | {score_std} | {val_mean} | {val_std} |".format(
                task=task_name,
                score_mean=f"{mean(scores):.6f}" if scores else "-",
                score_std=f"{population_std(scores):.6f}" if len(scores) > 1 else "-",
                val_mean=f"{mean(val_losses):.6f}" if len(val_losses) == 3 else "-",
                val_std=f"{population_std(val_losses):.6f}" if len(val_losses) == 3 else "-",
            )
        )

    if aggregate_lines:
        lines.extend(
            [
                "",
                "## Three-seed aggregate",
                "",
                "| task | mean best `test/mean_score` | std best `test/mean_score` | mean final `val_loss` | std final `val_loss` |",
                "| --- | ---: | ---: | ---: | ---: |",
                *aggregate_lines,
            ]
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def main() -> None:
    add_repo_paths()
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()

    rows = []
    for task_name in MIMICGEN_TASKS:
        config_name = task_config_name(task_name)
        task_dir = output_root / config_name
        if not task_dir.is_dir():
            continue
        for seed_dir in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name)
            except ValueError:
                continue
            row = summarize_run(seed_dir, config_name, seed)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: (row["task"], row["seed"]))
    report_root = reports_dir()
    write_csv(rows, report_root / "baseline_summary.csv")
    write_markdown(rows, report_root / "baseline_summary.md")
    print(f"Wrote {report_root / 'baseline_summary.csv'}")
    print(f"Wrote {report_root / 'baseline_summary.md'}")
    print("Next: python mydiffusion_D1/scripts/plot_results.py")


if __name__ == "__main__":
    main()

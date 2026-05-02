#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
    parser = argparse.ArgumentParser(description="Generate plots and rollout contact sheets for MimicGen D1 runs.")
    parser.add_argument("--output-dir", default=str(outputs_dir()), help="Training output root.")
    parser.add_argument("--reports-dir", default=str(reports_dir()), help="Report output root.")
    parser.add_argument("--task", action="append", default=None, help="Optional task config name filter.")
    parser.add_argument("--seed", action="append", type=int, default=None, help="Optional seed filter.")
    parser.add_argument("--max-videos", type=int, default=2, help="Max mp4 files to process per run.")
    parser.add_argument(
        "--frame-layout",
        choices=["fixed4"],
        default="fixed4",
        help="Frame extraction layout. Only fixed4 is supported in this version.",
    )
    parser.add_argument("--compare-task", default=None, help="Task config name for baseline vs AttnRes comparison.")
    parser.add_argument("--compare-seed", type=int, default=None, help="Seed for baseline vs AttnRes comparison.")
    parser.add_argument("--baseline-exp", default="baseline_tuned", help="Baseline experiment name for comparison mode.")
    parser.add_argument("--attnres-exp", default="attnres_tuned", help="AttnRes experiment name for comparison mode.")
    return parser.parse_args()


def figure_dirs(base_dir: Path) -> dict[str, Path]:
    root = base_dir / "figures"
    mapping = {
        "root": root,
        "curves": root / "curves",
        "aggregates": root / "aggregates",
        "contact_sheets": root / "contact_sheets",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def load_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def resolve_tasks(task_filters: list[str] | None) -> list[str]:
    all_tasks = [task_config_name(task) for task in MIMICGEN_TASKS]
    if not task_filters:
        return all_tasks
    invalid = sorted(set(task_filters) - set(all_tasks))
    if invalid:
        raise ValueError(f"Unsupported task filters: {invalid}. Available tasks: {all_tasks}")
    return [task for task in all_tasks if task in task_filters]


def parse_checkpoint_name(checkpoint_name: str) -> tuple[int, str, float] | None:
    match = CHECKPOINT_RE.match(checkpoint_name)
    if not match:
        return None
    metric_name = str(match.group("metric"))
    if metric_name not in CHECKPOINT_METRIC_MODES:
        return None
    return int(match.group("epoch")), metric_name, float(match.group("value"))


def _improves(metric_name: str, candidate_value: float, best_value: float | None, candidate_epoch: int, best_epoch: int) -> bool:
    if best_value is None:
        return True
    mode = CHECKPOINT_METRIC_MODES[metric_name]
    if mode == "max":
        return candidate_value > best_value or (candidate_value == best_value and candidate_epoch > best_epoch)
    return candidate_value < best_value or (candidate_value == best_value and candidate_epoch > best_epoch)


def discover_runs(output_root: Path, tasks: list[str], seed_filters: set[int] | None) -> list[dict]:
    runs: list[dict] = []
    for task_name in tasks:
        task_dir = output_root / task_name
        if not task_dir.is_dir():
            continue
        for child in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            try:
                seed = int(child.name)
            except ValueError:
                seed = None
            if seed is not None:
                if seed_filters is not None and seed not in seed_filters:
                    continue
                runs.append(
                    {
                        "task": task_name,
                        "exp_name": LEGACY_EXP_NAME,
                        "seed": seed,
                        "run_dir": child,
                        "log_path": child / "logs.json.txt",
                        "media_dirs": [child / "media", child / "rollout_export" / "media", child / "best_rollout_export" / "media"],
                    }
                )
                continue

            exp_name = child.name
            for seed_dir in sorted(child.iterdir(), key=lambda path: path.name):
                if not seed_dir.is_dir():
                    continue
                try:
                    nested_seed = int(seed_dir.name)
                except ValueError:
                    continue
                if seed_filters is not None and nested_seed not in seed_filters:
                    continue
                runs.append(
                    {
                        "task": task_name,
                        "exp_name": exp_name,
                        "seed": nested_seed,
                        "run_dir": seed_dir,
                        "log_path": seed_dir / "logs.json.txt",
                        "media_dirs": [
                            seed_dir / "media",
                            seed_dir / "rollout_export" / "media",
                            seed_dir / "best_rollout_export" / "media",
                        ],
                    }
                )
    return runs


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

    return {
        "task": task_name,
        "exp_name": exp_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
        "best_checkpoint_metric_value": best_checkpoint_metric_value,
        "best_test_mean_score": best_test_mean_score,
        "final_test_mean_score": final_test_mean_score,
        "final_val_loss": final_val_loss,
        "best_test_contact_rate": best_test_contact_rate,
        "best_test_mean_min_eef_object_distance": best_test_mean_min_eef_object_distance,
    }


def save_summary_csv(rows: list[dict], reports_root: Path) -> Path:
    path = reports_root / "baseline_summary.csv"
    fieldnames = [
        "task",
        "exp_name",
        "seed",
        "run_dir",
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
    return path


def load_or_build_summary(reports_root: Path, runs: list[dict]) -> list[dict]:
    csv_path = reports_root / "baseline_summary.csv"
    if csv_path.is_file():
        rows = []
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = dict(row)
                parsed["exp_name"] = parsed.get("exp_name") or LEGACY_EXP_NAME
                parsed["seed"] = int(parsed["seed"])
                for key in (
                    "best_checkpoint_metric_value",
                    "best_test_mean_score",
                    "final_test_mean_score",
                    "final_val_loss",
                    "best_test_contact_rate",
                    "best_test_mean_min_eef_object_distance",
                ):
                    value = parsed.get(key)
                    parsed[key] = None if value in (None, "", "-") else float(value)
                metric_name = parsed.get("best_checkpoint_metric_name")
                parsed["best_checkpoint_metric_name"] = None if metric_name in (None, "", "-") else metric_name
                rows.append(parsed)
        return rows

    rows: list[dict] = []
    for run in runs:
        summary = summarize_run(run["run_dir"], run["task"], run["exp_name"], run["seed"])
        if summary is not None:
            rows.append(summary)
    rows.sort(key=lambda row: (row["task"], row["exp_name"], row["seed"]))
    save_summary_csv(rows, reports_root)
    return rows


def _extract_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for index, row in enumerate(rows):
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        xs.append(float(row.get("epoch", row.get("global_step", index))))
        ys.append(float(value))
    if not xs:
        return np.array([]), np.array([])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _metric_title(metric_name: str | None) -> str:
    if metric_name == "test_mean_score":
        return "Best Test Mean Score"
    if metric_name == "val_loss":
        return "Best Checkpoint Val Loss"
    if metric_name == "test_mean_min_eef_object_distance":
        return "Best Min EEF-Object Distance"
    return "Best Checkpoint Metric"


def _resolve_accuracy_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, str] | None:
    for key, label in (
        ("test/contact_rate", "val_success_rate"),
        ("test/mean_score", "val_accuracy"),
        ("train/mean_score", "train_accuracy"),
    ):
        xs, ys = _extract_series(rows, key)
        if len(xs) > 0:
            return xs, ys, label
    return None


def _extract_run_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return _extract_series(rows, key)


def _load_rows_for_run(run_dir: Path) -> list[dict]:
    log_path = run_dir / "logs.json.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    rows = read_json_lines(log_path)
    if not rows:
        raise ValueError(f"No readable json lines in: {log_path}")
    return rows


def _find_run_dir_for_compare(output_root: Path, task: str, exp_name: str, seed: int) -> Path:
    candidate = output_root / task / exp_name
    if not candidate.is_dir():
        raise FileNotFoundError(f"Missing experiment directory: {candidate}")
    direct_run_dir = candidate / str(seed)
    if direct_run_dir.is_dir():
        return direct_run_dir

    nested_matches = []
    for variant_dir in sorted(candidate.iterdir(), key=lambda path: path.name):
        if not variant_dir.is_dir():
            continue
        run_dir = variant_dir / str(seed)
        if run_dir.is_dir():
            nested_matches.append(run_dir)

    if len(nested_matches) == 1:
        return nested_matches[0]
    if len(nested_matches) > 1:
        raise FileNotFoundError(
            f"Multiple run directories found for task={task}, exp_name={exp_name}, seed={seed}: "
            + ", ".join(str(path) for path in nested_matches)
        )

    raise FileNotFoundError(f"Missing run directory under {candidate} for seed={seed}")


def plot_baseline_vs_attnres(
    *,
    output_root: Path,
    reports_root: Path,
    task: str,
    seed: int,
    baseline_exp: str,
    attnres_exp: str,
) -> list[Path]:
    plt = load_matplotlib_pyplot()
    curves_dir = reports_root / "figures" / "comparisons"
    curves_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_dir = _find_run_dir_for_compare(output_root, task, baseline_exp, seed)
    attnres_run_dir = _find_run_dir_for_compare(output_root, task, attnres_exp, seed)
    baseline_rows = _load_rows_for_run(baseline_run_dir)
    attnres_rows = _load_rows_for_run(attnres_run_dir)

    saved: list[Path] = []

    # loss comparison
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    loss_series = []
    for rows, color, prefix in (
        (baseline_rows, "#2563eb", "Baseline"),
        (attnres_rows, "#ef4444", "AttnRes"),
    ):
        for key, suffix, linestyle in (
            ("train_loss", "Train Loss", "-"),
            ("val_loss", "Val Loss", "--"),
        ):
            xs, ys = _extract_run_series(rows, key)
            if len(xs) > 0:
                ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2.0, label=f"{prefix} {suffix}")
                loss_series.append((xs, ys))
    ax.set_title(f"Baseline vs AttnRes - {task} ({seed=}, 100 Epochs)".replace("seed=", "seed "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    if loss_series:
        max_epoch = int(max(float(np.max(xs)) for xs, _ in loss_series))
        inset_start = max(0, max_epoch - 20)
        inset = inset_axes(ax, width="38%", height="38%", loc="upper right", borderpad=2.0)
        min_y = float("inf")
        max_y = float("-inf")
        for rows, color, prefix in (
            (baseline_rows, "#2563eb", "Baseline"),
            (attnres_rows, "#ef4444", "AttnRes"),
        ):
            for key, linestyle in (
                ("train_loss", "-"),
                ("val_loss", "--"),
            ):
                xs, ys = _extract_run_series(rows, key)
                if len(xs) > 0:
                    inset.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.6)
                    mask = xs >= inset_start
                    if np.any(mask):
                        min_y = min(min_y, float(np.min(ys[mask])))
                        max_y = max(max_y, float(np.max(ys[mask])))
        inset.set_xlim(inset_start, max_epoch)
        if min_y < max_y:
            pad = max((max_y - min_y) * 0.15, 1e-4)
            inset.set_ylim(min_y - pad, max_y + pad)
        inset.set_title("Late Epochs", fontsize=9)
        inset.grid(alpha=0.25)
        inset.tick_params(labelsize=8)
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5")
    loss_path = curves_dir / f"{task}_seed{seed}_loss_comparison.png"
    fig.tight_layout()
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)
    saved.append(loss_path)

    # mse + accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    panel_specs = [
        ("train_action_mse_error", "Train MSE", "MSE"),
        ("val_action_mse_error", "Validation MSE", "MSE"),
    ]
    for axis, (metric_key, title, ylabel) in zip(axes[:2], panel_specs):
        for rows, color, prefix in (
            (baseline_rows, "#2563eb", "Baseline"),
            (attnres_rows, "#ef4444", "AttnRes"),
        ):
            xs, ys = _extract_run_series(rows, metric_key)
            if len(xs) > 0:
                axis.plot(xs, ys, color=color, linestyle="-", linewidth=2.0, label=prefix)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    acc_axis = axes[2]
    plotted_accuracy = False
    for rows, color, prefix in (
        (baseline_rows, "#2563eb", "Baseline"),
        (attnres_rows, "#ef4444", "AttnRes"),
    ):
        xs, ys = _extract_run_series(rows, "test/contact_rate")
        if len(xs) > 0:
            acc_axis.plot(xs, ys, color=color, linestyle="-", linewidth=2.0, label=prefix)
            plotted_accuracy = True
    acc_axis.set_title("Validation Success Rate")
    acc_axis.set_xlabel("Epoch")
    acc_axis.set_ylabel("Success Rate")
    acc_axis.set_ylim(0.0, 1.05)
    acc_axis.grid(alpha=0.3)
    if plotted_accuracy:
        acc_axis.legend(loc="best")
    else:
        acc_axis.text(
            0.5,
            0.5,
            "rollout metrics unavailable",
            ha="center",
            va="center",
            fontsize=12,
            transform=acc_axis.transAxes,
        )

    fig.suptitle(f"Baseline vs AttnRes - {task} (100 Epochs)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    mse_path = curves_dir / f"{task}_seed{seed}_mse_accuracy_comparison.png"
    fig.savefig(mse_path, dpi=180)
    plt.close(fig)
    saved.append(mse_path)

    return saved


def plot_run_curves(run: dict, curves_dir: Path) -> list[Path]:
    plt = load_matplotlib_pyplot()
    log_path = run["log_path"]
    if not log_path.is_file():
        return []

    rows = read_json_lines(log_path)
    if not rows:
        return []

    task = run["task"]
    exp_name = run["exp_name"]
    seed = run["seed"]
    base_name = f"{task}_{exp_name}_seed{seed}"
    saved: list[Path] = []

    has_loss_like = any(
        any(key in row for key in ("train_loss", "val_loss", "train_action_mse_error"))
        for row in rows
    )
    if has_loss_like:
        fig, ax = plt.subplots(figsize=(10.2, 5.8))
        plotted = False
        for key, label, color in (
            ("train_loss", "train_loss", "#1f77b4"),
            ("val_loss", "val_loss", "#ff7f0e"),
            ("train_action_mse_error", "train_mse", "#d62728"),
        ):
            xs, ys = _extract_series(rows, key)
            if len(xs) > 0:
                ax.plot(xs, ys, label=label, linewidth=1.8, color=color)
                plotted = True

        accuracy_series = _resolve_accuracy_series(rows)
        if accuracy_series is not None:
            xs, ys, label = accuracy_series
            ax.plot(xs, ys, label=label, linewidth=2.0, color="#2ca02c")
            plotted = True

        ax.set_title("Full Baseline Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.02)
        handles, labels = ax.get_legend_handles_labels()
        if plotted and handles:
            ax.legend(handles, labels, loc="center right", framealpha=0.92)
        merged_path = curves_dir / f"{base_name}_training_curve.png"
        fig.tight_layout()
        fig.savefig(merged_path, dpi=180)
        plt.close(fig)
        saved.append(merged_path)

        if accuracy_series is not None:
            xs, ys, label = accuracy_series
            fig, ax = plt.subplots(figsize=(8.0, 4.6))
            ax.plot(xs, ys, linewidth=2.0, color="#2ca02c", label=label)
            ax.set_title("Accuracy Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            accuracy_path = curves_dir / f"{base_name}_accuracy_curve.png"
            fig.tight_layout()
            fig.savefig(accuracy_path, dpi=180)
            plt.close(fig)
            saved.append(accuracy_path)

    score_keys = ("train/mean_score", "test/mean_score", "test/contact_rate", "test/mean_min_eef_object_distance")
    if any(any(key in row for key in score_keys) for row in rows):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (
            ("train/mean_score", "Train Mean Score"),
            ("test/mean_score", "Test Mean Score"),
            ("test/contact_rate", "Test Contact Rate"),
        ):
            xs, ys = _extract_series(rows, key)
            if len(xs) > 0:
                ax.plot(xs, ys, label=label, linewidth=1.6)
        ax.set_title(f"{task} {exp_name} seed {seed} score curves")
        ax.set_xlabel("epoch / global_step")
        ax.set_ylabel("metric")
        ax.grid(alpha=0.3)
        ax.legend()
        path = curves_dir / f"{base_name}_score_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

        xs, ys = _extract_series(rows, "test/mean_min_eef_object_distance")
        if len(xs) > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, ys, color="#2ca02c", linewidth=1.6)
            ax.set_title(f"{task} {exp_name} seed {seed} min eef-object distance")
            ax.set_xlabel("epoch / global_step")
            ax.set_ylabel("distance")
            ax.grid(alpha=0.3)
            path = curves_dir / f"{base_name}_distance_curve.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            saved.append(path)

    return saved


def plot_single_run_curves_for_dir(run_dir: Path, task: str, seed: int, curves_dir: Path, exp_name: str = LEGACY_EXP_NAME) -> list[Path]:
    curves_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "task": task,
        "exp_name": exp_name,
        "seed": seed,
        "run_dir": run_dir,
        "log_path": run_dir / "logs.json.txt",
        "media_dirs": [run_dir / "media", run_dir / "rollout_export" / "media", run_dir / "best_rollout_export" / "media"],
    }
    return plot_run_curves(run, curves_dir)


def plot_task_aggregate(task_name: str, rows: list[dict], aggregates_dir: Path) -> list[Path]:
    plt = load_matplotlib_pyplot()
    saved: list[Path] = []
    task_rows = [row for row in rows if row["task"] == task_name]
    if not task_rows:
        return saved

    exp_names = sorted({row["exp_name"] for row in task_rows})
    for exp_name in exp_names:
        exp_rows = sorted(
            [row for row in task_rows if row["exp_name"] == exp_name],
            key=lambda row: int(row["seed"]),
        )
        if not exp_rows:
            continue
        seeds = [str(row["seed"]) for row in exp_rows]
        best_metric_names = sorted(
            {
                row.get("best_checkpoint_metric_name")
                for row in exp_rows
                if row.get("best_checkpoint_metric_name") is not None and row.get("best_checkpoint_metric_value") is not None
            }
        )
        plot_best_metric = len(best_metric_names) == 1
        best_metric_name = best_metric_names[0] if plot_best_metric else None
        best_metric_values = [row.get("best_checkpoint_metric_value") for row in exp_rows]
        val_losses = [row["final_val_loss"] for row in exp_rows]

        subplot_count = 2 if plot_best_metric else 1
        fig, axes = plt.subplots(1, subplot_count, figsize=(10 if plot_best_metric else 5.6, 4.5))
        if subplot_count == 1:
            axes = [axes]

        axis_offset = 0
        if plot_best_metric:
            best_values = [float(value) for value in best_metric_values if value is not None]
            axes[0].bar(
                seeds,
                [0.0 if value is None else float(value) for value in best_metric_values],
                color="#1f77b4",
                alpha=0.85,
            )
            if best_values:
                best_mean = mean(best_values)
                best_std = population_std(best_values)
                axes[0].axhline(best_mean, color="#d62728", linestyle="--", linewidth=1.4, label=f"mean={best_mean:.3f}")
                if len(best_values) > 1:
                    axes[0].axhspan(best_mean - best_std, best_mean + best_std, color="#d62728", alpha=0.12, label=f"std={best_std:.3f}")
                axes[0].legend()
            axes[0].set_title(_metric_title(best_metric_name))
            axes[0].set_xlabel("seed")
            axes[0].set_ylabel("score" if best_metric_name == "test_mean_score" else "loss")
            axes[0].grid(axis="y", alpha=0.3)
            axis_offset = 1

        val_values = [float(value) for value in val_losses if value is not None]
        axes[axis_offset].bar(seeds, [0.0 if value is None else float(value) for value in val_losses], color="#2ca02c", alpha=0.85)
        if val_values:
            val_mean = mean(val_values)
            val_std = population_std(val_values)
            axes[axis_offset].axhline(val_mean, color="#9467bd", linestyle="--", linewidth=1.4, label=f"mean={val_mean:.3f}")
            if len(val_values) > 1:
                axes[axis_offset].axhspan(val_mean - val_std, val_mean + val_std, color="#9467bd", alpha=0.12, label=f"std={val_std:.3f}")
            axes[axis_offset].legend()
        axes[axis_offset].set_title("Final Val Loss")
        axes[axis_offset].set_xlabel("seed")
        axes[axis_offset].set_ylabel("loss")
        axes[axis_offset].grid(axis="y", alpha=0.3)

        fig.suptitle(f"{task_name} {exp_name} multi-seed summary")
        fig.tight_layout()
        path = aggregates_dir / f"{task_name}_{exp_name}_multiseed_summary.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_cross_task_bars(rows: list[dict], aggregates_dir: Path) -> list[Path]:
    plt = load_matplotlib_pyplot()
    exp_names = sorted({row["exp_name"] for row in rows})
    saved: list[Path] = []
    for exp_name in exp_names:
        filtered_rows = [row for row in rows if row["exp_name"] == exp_name]
        grouped = {task_config_name(task): [row for row in filtered_rows if row["task"] == task_config_name(task)] for task in MIMICGEN_TASKS}
        tasks = [task for task, task_rows in grouped.items() if task_rows]
        if not tasks:
            continue

        metric_names = {
            row.get("best_checkpoint_metric_name")
            for row in filtered_rows
            if row.get("best_checkpoint_metric_name") is not None and row.get("best_checkpoint_metric_value") is not None
        }
        plot_best_metric = len(metric_names) == 1
        best_metric_name = next(iter(metric_names)) if plot_best_metric else None
        best_metric_means = []
        best_metric_stds = []
        val_means = []
        val_stds = []
        for task in tasks:
            metric_values = [
                float(row["best_checkpoint_metric_value"])
                for row in grouped[task]
                if row.get("best_checkpoint_metric_name") == best_metric_name
                and row.get("best_checkpoint_metric_value") is not None
            ]
            val_values = [float(row["final_val_loss"]) for row in grouped[task] if row["final_val_loss"] is not None]
            best_metric_means.append(mean(metric_values) if metric_values else 0.0)
            best_metric_stds.append(population_std(metric_values) if len(metric_values) > 1 else 0.0)
            val_means.append(mean(val_values) if val_values else 0.0)
            val_stds.append(population_std(val_values) if len(val_values) > 1 else 0.0)

        x = np.arange(len(tasks))
        if plot_best_metric:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.bar(x, best_metric_means, yerr=best_metric_stds, capsize=4, color="#4c78a8", alpha=0.9)
            ax.set_xticks(x, tasks, rotation=12, ha="right")
            ax.set_ylabel("best checkpoint score" if best_metric_name == "test_mean_score" else "best checkpoint loss")
            ax.set_title(f"{exp_name}: D1 {_metric_title(best_metric_name).lower()} by task")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            metric_path = aggregates_dir / f"{exp_name}_all_tasks_best_checkpoint_metric_bar.png"
            fig.savefig(metric_path, dpi=180)
            plt.close(fig)
            saved.append(metric_path)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(x, val_means, yerr=val_stds, capsize=4, color="#59a14f", alpha=0.9)
        ax.set_xticks(x, tasks, rotation=12, ha="right")
        ax.set_ylabel("final val_loss")
        ax.set_title(f"{exp_name}: D1 final val loss by task")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        val_path = aggregates_dir / f"{exp_name}_all_tasks_final_val_loss_bar.png"
        fig.savefig(val_path, dpi=180)
        plt.close(fig)
        saved.append(val_path)

    return saved


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def select_fixed4_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    if len(frames) == 1:
        return [frames[0]] * 4
    indices = sorted({0, max(0, len(frames) // 3), max(0, (2 * len(frames)) // 3), len(frames) - 1})
    selected = [frames[index] for index in indices]
    while len(selected) < 4:
        selected.append(selected[-1])
    return selected[:4]


def build_contact_sheet(selected_frames: list[np.ndarray], output_path: Path, title: str) -> Path | None:
    plt = load_matplotlib_pyplot()
    if not selected_frames:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    labels = ["start", "1/3", "2/3", "end"]
    for axis, frame, label in zip(axes, selected_frames, labels):
        axis.imshow(frame)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_contact_sheets(run: dict, contact_dir: Path, max_videos: int, frame_layout: str) -> list[Path]:
    videos: list[Path] = []
    for media_dir in run["media_dirs"]:
        if media_dir.is_dir():
            videos.extend(sorted(media_dir.glob("*.mp4")))
    videos = sorted({video.resolve() for video in videos})
    if not videos:
        return []

    task = run["task"]
    exp_name = run["exp_name"]
    seed = run["seed"]
    base_name = f"{task}_{exp_name}_seed{seed}"
    saved: list[Path] = []
    for index, video_path in enumerate(videos[: max(1, max_videos)]):
        frames = read_video_frames(video_path)
        selected = select_fixed4_frames(frames) if frame_layout == "fixed4" else select_fixed4_frames(frames)
        suffix = "" if index == 0 else f"_{index + 1:02d}"
        output_path = contact_dir / f"{base_name}_rollout_contact_sheet{suffix}.png"
        result = build_contact_sheet(
            selected_frames=selected,
            output_path=output_path,
            title=f"{task} {exp_name} seed {seed} rollout{suffix or ' #1'}",
        )
        if result is not None:
            saved.append(result)
    return saved


def main() -> None:
    add_repo_paths()
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    reports_root = Path(args.reports_dir).expanduser().resolve()
    dirs = figure_dirs(reports_root)

    if args.compare_task and args.compare_seed is not None:
        compare_paths = plot_baseline_vs_attnres(
            output_root=output_root,
            reports_root=reports_root,
            task=args.compare_task,
            seed=int(args.compare_seed),
            baseline_exp=str(args.baseline_exp),
            attnres_exp=str(args.attnres_exp),
        )
        print(f"Wrote {len(compare_paths)} comparison figure(s)")
        for path in compare_paths:
            print(path)
        return

    tasks = resolve_tasks(args.task)
    seed_filters = set(args.seed) if args.seed else None
    runs = discover_runs(output_root=output_root, tasks=tasks, seed_filters=seed_filters)
    if not runs:
        raise FileNotFoundError(f"No runs found under {output_root} for tasks={tasks} seeds={sorted(seed_filters) if seed_filters else 'all'}.")

    summary_rows = load_or_build_summary(reports_root=reports_root, runs=runs)

    generated: list[Path] = []
    for run in runs:
        generated.extend(plot_run_curves(run, dirs["curves"]))
        generated.extend(plot_contact_sheets(run, dirs["contact_sheets"], max_videos=args.max_videos, frame_layout=args.frame_layout))

    for task in tasks:
        generated.extend(plot_task_aggregate(task, summary_rows, dirs["aggregates"]))
    generated.extend(plot_cross_task_bars(summary_rows, dirs["aggregates"]))

    print(f"Wrote {len(generated)} figure(s) under {dirs['root']}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

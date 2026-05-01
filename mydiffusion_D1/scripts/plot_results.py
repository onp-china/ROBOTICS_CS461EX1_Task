#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

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


def discover_runs(output_root: Path, tasks: list[str], seed_filters: set[int] | None) -> list[dict]:
    runs: list[dict] = []
    for task_name in tasks:
        task_dir = output_root / task_name
        if not task_dir.is_dir():
            continue
        for seed_dir in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name)
            except ValueError:
                continue
            if seed_filters is not None and seed not in seed_filters:
                continue
            runs.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "run_dir": seed_dir,
                    "log_path": seed_dir / "logs.json.txt",
                    "media_dir": seed_dir / "media",
                }
            )
    return runs


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

    return {
        "task": task_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_test_mean_score": best_test_mean_score,
        "final_test_mean_score": final_test_mean_score,
        "final_val_loss": final_val_loss,
    }


def save_summary_csv(rows: list[dict], reports_root: Path) -> Path:
    path = reports_root / "baseline_summary.csv"
    fieldnames = [
        "task",
        "seed",
        "run_dir",
        "best_test_mean_score",
        "final_test_mean_score",
        "final_val_loss",
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
                parsed["seed"] = int(parsed["seed"])
                for key in ("best_test_mean_score", "final_test_mean_score", "final_val_loss"):
                    value = parsed.get(key)
                    parsed[key] = None if value in (None, "", "-") else float(value)
                rows.append(parsed)
        return rows

    rows: list[dict] = []
    for run in runs:
        summary = summarize_run(run["run_dir"], run["task"], run["seed"])
        if summary is not None:
            rows.append(summary)
    rows.sort(key=lambda row: (row["task"], row["seed"]))
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


def plot_run_curves(run: dict, curves_dir: Path) -> list[Path]:
    plt = load_matplotlib_pyplot()
    log_path = run["log_path"]
    if not log_path.is_file():
        return []

    rows = read_json_lines(log_path)
    if not rows:
        return []

    task = run["task"]
    seed = run["seed"]
    saved: list[Path] = []

    if any(any(key in row for key in ("train_loss", "val_loss")) for row in rows):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (("train_loss", "Train Loss"), ("val_loss", "Val Loss")):
            xs, ys = _extract_series(rows, key)
            if len(xs) > 0:
                ax.plot(xs, ys, label=label, linewidth=1.6)
        ax.set_title(f"{task} seed {seed} loss curves")
        ax.set_xlabel("epoch / global_step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
        path = curves_dir / f"{task}_seed{seed}_loss_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    if any("train_action_mse_error" in row for row in rows):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs, ys = _extract_series(rows, "train_action_mse_error")
        if len(xs) > 0:
            ax.plot(xs, ys, color="#d95f02", linewidth=1.6)
        ax.set_title(f"{task} seed {seed} train action MSE")
        ax.set_xlabel("epoch / global_step")
        ax.set_ylabel("mse")
        ax.grid(alpha=0.3)
        path = curves_dir / f"{task}_seed{seed}_action_mse_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    if any(any(key in row for key in ("train/mean_score", "test/mean_score")) for row in rows):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (("train/mean_score", "Train Mean Score"), ("test/mean_score", "Test Mean Score")):
            xs, ys = _extract_series(rows, key)
            if len(xs) > 0:
                ax.plot(xs, ys, label=label, linewidth=1.6)
        ax.set_title(f"{task} seed {seed} score curves")
        ax.set_xlabel("epoch / global_step")
        ax.set_ylabel("mean score")
        ax.grid(alpha=0.3)
        ax.legend()
        path = curves_dir / f"{task}_seed{seed}_score_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_task_aggregate(task_name: str, rows: list[dict], aggregates_dir: Path) -> Path | None:
    plt = load_matplotlib_pyplot()
    task_rows = [row for row in rows if row["task"] == task_name]
    if not task_rows:
        return None

    task_rows = sorted(task_rows, key=lambda row: int(row["seed"]))
    seeds = [str(row["seed"]) for row in task_rows]
    scores = [row["best_test_mean_score"] for row in task_rows]
    val_losses = [row["final_val_loss"] for row in task_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    score_values = [float(value) for value in scores if value is not None]
    axes[0].bar(seeds, [0.0 if value is None else float(value) for value in scores], color="#1f77b4", alpha=0.85)
    if score_values:
        score_mean = mean(score_values)
        score_std = population_std(score_values)
        axes[0].axhline(score_mean, color="#d62728", linestyle="--", linewidth=1.4, label=f"mean={score_mean:.3f}")
        if len(score_values) > 1:
            axes[0].axhspan(score_mean - score_std, score_mean + score_std, color="#d62728", alpha=0.12, label=f"std={score_std:.3f}")
        axes[0].legend()
    axes[0].set_title("Best Test Mean Score")
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.3)

    val_values = [float(value) for value in val_losses if value is not None]
    axes[1].bar(seeds, [0.0 if value is None else float(value) for value in val_losses], color="#2ca02c", alpha=0.85)
    if val_values:
        val_mean = mean(val_values)
        val_std = population_std(val_values)
        axes[1].axhline(val_mean, color="#9467bd", linestyle="--", linewidth=1.4, label=f"mean={val_mean:.3f}")
        if len(val_values) > 1:
            axes[1].axhspan(val_mean - val_std, val_mean + val_std, color="#9467bd", alpha=0.12, label=f"std={val_std:.3f}")
        axes[1].legend()
    axes[1].set_title("Final Val Loss")
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("loss")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"{task_name} multi-seed summary")
    fig.tight_layout()
    path = aggregates_dir / f"{task_name}_multiseed_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cross_task_bars(rows: list[dict], aggregates_dir: Path) -> list[Path]:
    plt = load_matplotlib_pyplot()
    grouped = {}
    for task in [task_config_name(task) for task in MIMICGEN_TASKS]:
        grouped[task] = [row for row in rows if row["task"] == task]

    tasks = [task for task, task_rows in grouped.items() if task_rows]
    if not tasks:
        return []

    score_means = []
    score_stds = []
    val_means = []
    val_stds = []
    for task in tasks:
        score_values = [float(row["best_test_mean_score"]) for row in grouped[task] if row["best_test_mean_score"] is not None]
        val_values = [float(row["final_val_loss"]) for row in grouped[task] if row["final_val_loss"] is not None]
        score_means.append(mean(score_values) if score_values else 0.0)
        score_stds.append(population_std(score_values) if len(score_values) > 1 else 0.0)
        val_means.append(mean(val_values) if val_values else 0.0)
        val_stds.append(population_std(val_values) if len(val_values) > 1 else 0.0)

    x = np.arange(len(tasks))
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, score_means, yerr=score_stds, capsize=4, color="#4c78a8", alpha=0.9)
    ax.set_xticks(x, tasks, rotation=12, ha="right")
    ax.set_ylabel("best test/mean_score")
    ax.set_title("D1 best score by task")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    score_path = aggregates_dir / "all_tasks_best_score_bar.png"
    fig.savefig(score_path, dpi=180)
    plt.close(fig)
    saved.append(score_path)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, val_means, yerr=val_stds, capsize=4, color="#59a14f", alpha=0.9)
    ax.set_xticks(x, tasks, rotation=12, ha="right")
    ax.set_ylabel("final val_loss")
    ax.set_title("D1 final val loss by task")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    val_path = aggregates_dir / "all_tasks_final_val_loss_bar.png"
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
    indices = sorted(
        {
            0,
            max(0, len(frames) // 3),
            max(0, (2 * len(frames)) // 3),
            len(frames) - 1,
        }
    )
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
    media_dir = run["media_dir"]
    if not media_dir.is_dir():
        return []
    videos = sorted(media_dir.glob("*.mp4"))
    if not videos:
        return []

    task = run["task"]
    seed = run["seed"]
    saved: list[Path] = []
    for index, video_path in enumerate(videos[: max(1, max_videos)]):
        frames = read_video_frames(video_path)
        selected = select_fixed4_frames(frames) if frame_layout == "fixed4" else select_fixed4_frames(frames)
        suffix = "" if index == 0 else f"_{index + 1:02d}"
        output_path = contact_dir / f"{task}_seed{seed}_rollout_contact_sheet{suffix}.png"
        result = build_contact_sheet(
            selected_frames=selected,
            output_path=output_path,
            title=f"{task} seed {seed} rollout{suffix or ' #1'}",
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
        path = plot_task_aggregate(task, summary_rows, dirs["aggregates"])
        if path is not None:
            generated.append(path)
    generated.extend(plot_cross_task_bars(summary_rows, dirs["aggregates"]))

    print(f"Wrote {len(generated)} figure(s) under {dirs['root']}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

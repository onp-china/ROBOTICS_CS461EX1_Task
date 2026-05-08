#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker


def iter_json_objects_from_line(line: str):
    decoder = json.JSONDecoder()
    idx = 0
    n = len(line)
    while idx < n:
        brace_idx = line.find("{", idx)
        if brace_idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(line, brace_idx)
        except json.JSONDecodeError:
            idx = brace_idx + 1
            continue
        yield obj
        idx = end


def load_records(log_path: Path):
    records = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for obj in iter_json_objects_from_line(line):
                if "global_step" in obj and "epoch" in obj:
                    records.append(obj)
    return records


def split_sessions(records):
    sessions = []
    cur = []
    prev_epoch = None
    for rec in records:
        epoch = rec.get("epoch")
        if cur and epoch == 0 and (prev_epoch is not None) and prev_epoch > 0:
            sessions.append(cur)
            cur = []
        cur.append(rec)
        prev_epoch = epoch
    if cur:
        sessions.append(cur)
    return sessions


def sanitize_metric_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def plot_metric(session, metric: str, out_path: Path):
    xs = []
    ys = []
    for rec in session:
        if metric in rec:
            xs.append(rec["global_step"])
            ys.append(rec[metric])
    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, ys, linewidth=1.1)

    if metric == "lr":
        # 让坐标轴明确标出“1e-4 基准 + 1e-8 级别变化”
        ax.set_ylabel("learning rate")
        ax.set_title("learning rate (absolute) vs global_step")
        # 关键：打开 offset，并强制用科学计数法显示刻度的数量级
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-8, -8), useOffset=True)
        # 美化：用科学计数法显示 offset 文本
        sf = ticker.ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-8, -8))
        sf.set_useOffset(True)
        ax.yaxis.set_major_formatter(sf)
        ax.tick_params(axis="y", labelsize=9)
    else:
        ax.set_ylabel(metric)
        ax.set_title(metric)

    ax.set_xlabel("global_step")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from latest training session in logs.json.txt")
    parser.add_argument("--log", required=True, help="Path to logs.json.txt")
    parser.add_argument("--out-dir", required=True, help="Output directory for png figures")
    parser.add_argument("--end-step", type=int, default=None, help="Optional: keep records with global_step <= end-step")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(log_path)
    if not records:
        raise RuntimeError(f"No records found in {log_path}")
    if args.end_step is not None:
        records = [r for r in records if r["global_step"] <= args.end_step]
        if not records:
            raise RuntimeError(f"No records found after applying end-step={args.end_step}")
    sessions = split_sessions(records)
    session = sessions[-1]

    metrics = [
        "train_loss",
        "val_loss",
        "train_action_mse_error",
        "lr",
        "train/mean_score",
        "test/mean_score",
        "train/sim_max_reward_0",
        "train/sim_max_reward_1",
        "test/sim_max_reward_100000",
        "test/sim_max_reward_100001",
        "test/sim_max_reward_100002",
        "test/sim_max_reward_100003",
    ]

    written = []
    for metric in metrics:
        out_path = out_dir / f"{sanitize_metric_name(metric)}.png"
        if plot_metric(session, metric, out_path):
            written.append(out_path.name)

    summary = {
        "n_total_records": len(records),
        "n_sessions": len(sessions),
        "latest_session_records": len(session),
        "latest_session_epoch_min": min(r["epoch"] for r in session),
        "latest_session_epoch_max": max(r["epoch"] for r in session),
        "latest_session_step_min": min(r["global_step"] for r in session),
        "latest_session_step_max": max(r["global_step"] for r in session),
        "written_figures": written,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

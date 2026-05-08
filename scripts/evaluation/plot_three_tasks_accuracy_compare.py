#!/usr/bin/env python3
"""
1×3 panels: MimicGen rollout success rate (test/mean_score) vs epoch.
Baseline = TransformerForDiffusion; optional ResAttention log paths.

Three-piece assembly panel can plot two RA curves (run1 and run2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(log_path: Path) -> list[dict]:
    records: list[dict] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "global_step" in obj and "epoch" in obj:
                records.append(obj)
    return records


def split_sessions(records: list[dict]) -> list[list[dict]]:
    sessions: list[list[dict]] = []
    cur: list[dict] = []
    prev_epoch: int | None = None
    for rec in records:
        epoch = rec.get("epoch")
        if cur and epoch == 0 and prev_epoch is not None and prev_epoch > 0:
            sessions.append(cur)
            cur = []
        cur.append(rec)
        prev_epoch = epoch
    if cur:
        sessions.append(cur)
    return sessions


def rollout_series(session: list[dict]) -> tuple[list[int], list[float]]:
    """Last rollout record per epoch (dedup by epoch)."""
    by_epoch: dict[int, float] = {}
    for rec in session:
        if "test/mean_score" not in rec:
            continue
        by_epoch[int(rec["epoch"])] = float(rec["test/mean_score"])
    epochs = sorted(by_epoch.keys())
    scores = [by_epoch[e] for e in epochs]
    return epochs, scores


def summarize_rollout(session: list[dict]) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    """Return (last, best) as (epoch, score)."""
    epochs, scores = rollout_series(session)
    if not epochs:
        return None, None
    last = (epochs[-1], float(scores[-1]))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = (epochs[best_idx], float(scores[best_idx]))
    return last, best


def annotate_compare_box(
    ax,
    *,
    baseline_sess: list[dict] | None,
    ra_entries: list[tuple[str, list[dict]]],
) -> None:
    """Top-left box: best rollout score per curve."""
    lines: list[str] = []
    if baseline_sess:
        _, best_b = summarize_rollout(baseline_sess)
        if best_b is not None:
            lines.append(f"Baseline best: ep {best_b[0]}, score {best_b[1]:.2f}")
    for label, sess in ra_entries:
        _, best_r = summarize_rollout(sess)
        if best_r is not None:
            lines.append(f"{label} best: ep {best_r[0]}, score {best_r[1]:.2f}")
    if not lines:
        return
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=7.8,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbd5e1", alpha=0.94),
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Three MimicGen tasks: baseline vs ResAttention (test/mean_score curves)."
    )
    parser.add_argument(
        "--out",
        default=str(repo / "outputs" / "figures" / "accuracy_three_tasks_baseline_vs_resattention.png"),
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=160)
    # baseline（初版）默认路径 seed=42
    parser.add_argument(
        "--baseline-coffee",
        default=str(repo / "outputs" / "coffee_d0_lowdim_abs" / "42" / "logs.json.txt"),
    )
    parser.add_argument(
        "--baseline-mug",
        default=str(repo / "outputs" / "mug_cleanup_d0_lowdim_abs" / "42" / "logs.json.txt"),
    )
    parser.add_argument(
        "--baseline-three-piece",
        default=str(repo / "outputs" / "three_piece_assembly_d0_lowdim_abs" / "42" / "logs.json.txt"),
    )
    # ResAttention（可选；不存在则只画初版）
    parser.add_argument(
        "--ra-coffee",
        default=str(repo / "outputs_resattention" / "coffee_d0_ra" / "logs.json.txt"),
        help="ResAttention coffee logs.json.txt (skipped if missing)",
    )
    parser.add_argument(
        "--ra-mug",
        default=str(repo / "outputs_resattention" / "mug_cleanup_d0_ra" / "logs.json.txt"),
        help="ResAttention mug_cleanup logs.json.txt (skipped if missing)",
    )
    parser.add_argument(
        "--ra-three-piece",
        default=str(repo / "outputs_resattention" / "run1" / "logs.json.txt"),
        help="ResAttention three-piece assembly run1 logs.json.txt",
    )
    parser.add_argument(
        "--ra-three-piece-run2",
        default=str(repo / "outputs_resattention" / "run2" / "logs.json.txt"),
        help="ResAttention three-piece assembly run2 logs.json.txt (skipped if missing)",
    )
    args = parser.parse_args()

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    tasks: list[tuple[str, str, str]] = [
        ("Coffee (d0)", args.baseline_coffee, args.ra_coffee),
        ("Mug cleanup (d0)", args.baseline_mug, args.ra_mug),
        ("Three-piece assembly (d0)", args.baseline_three_piece, args.ra_three_piece),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.85), sharey=True)
    color_base = "#2563eb"
    color_ra1 = "#dc2626"
    color_ra2 = "#16a34a"

    for idx, (ax, (title, base_path, ra_path)) in enumerate(zip(axes, tasks)):
        baseline_sess: list[dict] | None = None
        ra_for_notes: list[tuple[str, list[dict]]] = []

        bp = Path(base_path)
        if bp.is_file():
            baseline_sess = split_sessions(load_records(bp))[-1]
            ex, sy = rollout_series(baseline_sess)
            if ex:
                ax.plot(ex, sy, color=color_base, lw=1.4, label="Baseline Transformer")
        else:
            ax.text(
                0.5,
                0.5,
                "No baseline log",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

        if idx == 2:
            # 三块装配：run1 + run2 两条 ResAttention
            ra_specs: list[tuple[str, str, str]] = [
                (args.ra_three_piece, "ResAttention (run1)", color_ra1),
                (args.ra_three_piece_run2, "ResAttention run2", color_ra2),
            ]
            any_ra = False
            for path_str, legend_label, c in ra_specs:
                ra_p = Path(path_str) if path_str else Path("")
                if ra_p.is_file():
                    sess_ra = split_sessions(load_records(ra_p))[-1]
                    exr, syr = rollout_series(sess_ra)
                    if exr:
                        ax.plot(exr, syr, color=c, lw=1.4, label=legend_label)
                        any_ra = True
                        short = "RA run1" if "run1" in legend_label else "RA run2"
                        ra_for_notes.append((short, sess_ra))
            if not any_ra:
                ax.text(
                    0.98,
                    0.06,
                    "(no ResAttention log)",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#64748b",
                )
            annotate_compare_box(ax, baseline_sess=baseline_sess, ra_entries=ra_for_notes)
        else:
            ra_p = Path(ra_path) if ra_path else Path("")
            if ra_p.is_file():
                sess_ra = split_sessions(load_records(ra_p))[-1]
                exr, syr = rollout_series(sess_ra)
                if exr:
                    ax.plot(exr, syr, color=color_ra1, lw=1.4, label="ResAttention")
                    ra_for_notes.append(("RA", sess_ra))
            elif ra_path:
                ax.text(
                    0.98,
                    0.06,
                    "(no ResAttention log)",
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#64748b",
                )
            annotate_compare_box(ax, baseline_sess=baseline_sess, ra_entries=ra_for_notes)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.28)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    axes[0].set_ylabel("test / mean_score (success rate)")

    fig.suptitle(
        "MimicGen: Baseline vs ResAttention (coffee & mug RA; three-piece run1 & run2)",
        fontsize=11.8,
        y=1.02,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

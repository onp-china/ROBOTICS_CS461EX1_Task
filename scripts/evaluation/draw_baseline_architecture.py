#!/usr/bin/env python3
"""
Draw baseline architecture diagram for Diffusion Policy (Transformer backbone)
including diffusion training, attention module, and closed-loop RL interaction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, body, fc, ec="#1f2937", title_size=11, body_size=9):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.8",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + 0.8, y + h - 1.2, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color="#0f172a")
    ax.text(x + 0.8, y + h - 3.2, body, ha="left", va="top", fontsize=body_size, color="#0f172a", linespacing=1.35)


def add_arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.8, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        linestyle=ls,
        color=color,
    )
    ax.add_patch(arr)


def main() -> None:
    out_path = Path("/Users/kiki/Desktop/mydiffusion/outputs/figures/baseline_diffusion_rl_attention.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    # Title
    ax.text(
        2,
        68,
        "Baseline Architecture (Paper-Aligned): Diffusion + Attention + Closed-Loop RL Interaction",
        fontsize=16,
        fontweight="bold",
        color="#0b132b",
        ha="left",
        va="top",
    )
    ax.text(
        2,
        65.5,
        "Based on Diffusion Policy transformer formulation: conditional action diffusion, causal self-attention, cross-attention conditioning, receding-horizon execution.",
        fontsize=10.5,
        color="#334155",
        ha="left",
        va="top",
    )

    # Section backgrounds
    add_box(
        ax,
        2,
        37,
        96,
        26,
        "A) Offline Training (Behavior Cloning with Diffusion Loss)",
        "Training target is denoising error, not policy gradient.",
        fc="#eef4ff",
        ec="#6b8afd",
        title_size=12,
        body_size=9,
    )
    add_box(
        ax,
        2,
        3,
        96,
        30,
        "B) Inference + Closed-Loop Control (RL-style Environment Interaction)",
        "Execute first action chunk, re-observe, and re-plan (receding horizon).",
        fc="#ecfdf3",
        ec="#34c759",
        title_size=12,
        body_size=9,
    )

    # Training pipeline boxes
    add_box(
        ax,
        5,
        48,
        20,
        12,
        "Demonstration Batch",
        "Sample (O_t, A_t^0)\nO_t: latest To observations\nA_t^0: clean action chunk",
        fc="#ffffff",
    )

    add_box(
        ax,
        29,
        48,
        20,
        12,
        "Forward Diffusion",
        "Choose k and epsilon~N(0,I)\nA_t^k = sqrt(alpha_bar_k) A_t^0\n      + sqrt(1-alpha_bar_k) epsilon",
        fc="#ffffff",
    )

    add_box(
        ax,
        53,
        44,
        24,
        20,
        "Noise Predictor epsilon_theta",
        "Time-Series Diffusion Transformer\ninput: A_t^k, O_t, k\noutput: predicted noise",
        fc="#fffdf4",
        ec="#f0b429",
    )

    add_box(
        ax,
        81,
        48,
        14,
        12,
        "Training Loss",
        "L = E[||epsilon -\n      epsilon_theta(A_t^k,O_t,k)||^2]",
        fc="#ffffff",
    )

    # Attention sub-structure inside predictor
    add_box(
        ax,
        55,
        57,
        20,
        5.2,
        "Causal Self-Attention",
        "action tokens attend only to past/current tokens",
        fc="#fff7d6",
        ec="#d4a017",
        title_size=9,
        body_size=7.6,
    )
    add_box(
        ax,
        55,
        50.9,
        20,
        5.2,
        "Cross-Attention Conditioning",
        "query: action tokens; key/value: obs embedding + step embedding",
        fc="#ffecc9",
        ec="#d4a017",
        title_size=9,
        body_size=7.4,
    )
    add_box(
        ax,
        55,
        44.8,
        20,
        5.2,
        "FFN + Projection",
        "per-token output predicts denoising direction",
        fc="#fff7d6",
        ec="#d4a017",
        title_size=9,
        body_size=7.6,
    )

    # Training arrows
    add_arrow(ax, 25, 54, 29, 54)
    add_arrow(ax, 49, 54, 53, 54)
    add_arrow(ax, 77, 54, 81, 54)

    # Inference / control loop
    add_box(
        ax,
        5,
        19,
        22,
        11,
        "Observation Window",
        "Collect latest To observations\nfrom environment state",
        fc="#ffffff",
    )

    add_box(
        ax,
        31,
        16,
        32,
        17,
        "Diffusion Inference",
        "Initialize A_t^K ~ N(0,I)\nfor k=K...1:\n  A_t^{k-1} <- reverse_step(\n    A_t^k, epsilon_theta(A_t^k,O_t,k))\nOutput denoised action sequence A_t^0",
        fc="#ffffff",
        ec="#2f6df6",
    )

    add_box(
        ax,
        67,
        19,
        14,
        11,
        "Receding Horizon",
        "Execute first Ta actions\n(remaining actions dropped)\nthen re-plan at t+Ta",
        fc="#ffffff",
    )

    add_box(
        ax,
        84,
        15,
        12,
        19,
        "Robot Env",
        "Dynamics + rewards\n(success/score)\n\nUsed for rollout\nevaluation",
        fc="#ffffff",
        ec="#2b8a3e",
    )

    # Loop arrows
    add_arrow(ax, 27, 24.5, 31, 24.5)
    add_arrow(ax, 63, 24.5, 67, 24.5)
    add_arrow(ax, 81, 24.5, 84, 24.5)
    add_arrow(ax, 84, 20.0, 27, 20.0, color="#2b8a3e")

    # Dashed note for RL interpretation
    add_box(
        ax,
        31,
        6,
        48,
        7,
        "RL Relation (for this baseline)",
        "Policy is trained offline from demonstrations (BC-style diffusion loss).\nEnvironment rewards/success are used for closed-loop rollout evaluation, not policy-gradient updates.",
        fc="#f8fffb",
        ec="#38a169",
        title_size=10,
        body_size=8.5,
    )

    # Baseline hyperparameter note from this repo
    add_box(
        ax,
        82,
        39,
        14,
        7,
        "This Repo Baseline",
        "horizon=16\nobs steps=2\naction steps=8\n8 layers, 4 heads",
        fc="#f8fafc",
        ec="#64748b",
        title_size=9.2,
        body_size=8.0,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()

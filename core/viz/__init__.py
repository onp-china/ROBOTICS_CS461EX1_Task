"""core.viz: 模型架构可视化工具（文本摘要、matplotlib 架构图、Mermaid 流程图）。"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn


__all__ = ["summary", "draw_model", "render_mermaid"]


# ---------------------------------------------------------------------------
# 1. summary — 文本摘要
# ---------------------------------------------------------------------------

def summary(model: nn.Module, sample_shape: tuple, cond_shape: tuple) -> dict[str, Any]:
    """
    打印并返回每层的 shape + 参数量统计。

    Parameters
    ----------
    model : nn.Module
    sample_shape : (B, T, input_dim)
    cond_shape    : (B, n_obs_steps, cond_dim)

    Returns
    -------
    dict with keys: ``total_params``, ``trainable_params``, ``layers``
    """
    layers = []
    total = 0
    trainable = 0

    print(f"\n{'='*60}")
    print(f" Model: {model.__class__.__name__}")
    print(f" sample shape: {sample_shape}  |  cond shape: {cond_shape}")
    print(f"{'='*60}")
    print(f"{'Module':<40} {'Output':>20} {'Params':>12}")
    print("-" * 74)

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # 跳过容器，只打印叶子模块
        params = sum(p.numel() for p in module.parameters())
        if params == 0:
            continue
        total += params
        trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)

        # 估算输出 shape（近似）
        out = _leaf_output_shape(module, sample_shape, cond_shape)
        print(f"  {name:<38} {str(out):>20} {params:>12,}")
        layers.append(dict(name=name, output_shape=out, n_params=params))

    print("-" * 74)
    print(f"  {'TOTAL':<38} {'':>20} {total:>12,}")
    print(f"  {'TRAINABLE':<38} {'':>20} {trainable:>12,}")
    print(f"{'='*60}\n")

    return dict(total_params=total, trainable_params=trainable, layers=layers)


def _leaf_output_shape(module: nn.Module, sample_shape, cond_shape):
    """为常见层类型估算输出 shape（粗略）。"""
    import torch
    name = module.__class__.__name__
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        return f"({in_features},) → ({out_features},)"
    if isinstance(module, (nn.MultiheadAttention,)):
        embed = module.embed_dim
        return f"(..., {embed})"
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        return f"({module.normalized_shape},)"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return f"Conv{module.__class__.__name__[-2:]} out_channels={module.out_channels}"
    if isinstance(module, nn.Sequential):
        return "Sequential(...)"
    if hasattr(module, 'get_output_shape'):
        return module.get_output_shape()
    return name


# ---------------------------------------------------------------------------
# 2. draw_model — matplotlib 论文风格架构图
# ---------------------------------------------------------------------------

def draw_model(model: nn.Module, sample_shape: tuple, cond_shape: tuple, figsize=(14, 10)):
    """
    用 matplotlib 画一个简化架构图：Embedding → Encoder → Decoder → Head。
    每个模块用一个彩色 box 表示，内部标注类型和参数量级。
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(f"Model Architecture — {model.__class__.__name__}", fontsize=14, pad=12)

    # 手动定义几个 stage
    B, T, I = sample_shape
    _, n_obs, C = cond_shape
    stages = [
        ("Input\nEmbedding", f"x∈ℝ^{I}\ncond∈ℝ^{C}", "#e3f2fd"),
        ("Positional\nEmbedding", f"T={T}", "#fff9c4"),
        ("Time & Cond\nEncoder", f"n_obs={n_obs}", "#f3e5f5"),
        ("AttnRes\nDecoder Stack", f"n_layer={getattr(model, 'horizon', '?')}", "#e8f5e9"),
        ("Final\nLayerNorm", "LayerNorm", "#fce4ec"),
        ("Output\nHead", f"out∈ℝ^{I}", "#fff3e0"),
    ]

    n = len(stages)
    xs = [1.0 + i * (8.0 / (n - 1)) for i in range(n)]
    y_center = 5.5

    for i, ((label, detail, color), x) in enumerate(zip(stages, xs)):
        box = mpatches.FancyBboxPatch(
            (x - 0.6, y_center - 0.9), 1.2, 1.8,
            boxstyle="round,pad=0.05", fc=color, ec="#555", lw=1.2
        )
        ax.add_patch(box)
        ax.text(x, y_center, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x, y_center - 1.2, detail, ha="center", va="center", fontsize=6.5, color="#444")

        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i + 1] - 0.6, y_center),
                xytext=(x + 0.6, y_center),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
            )

    # legend
    legend_items = [
        mpatches.Patch(fc="#e3f2fd", label="Input / Embedding"),
        mpatches.Patch(fc="#fff9c4", label="Positional Encoding"),
        mpatches.Patch(fc="#f3e5f5", label="Condition Encoder"),
        mpatches.Patch(fc="#e8f5e9", label="AttnRes Decoder"),
        mpatches.Patch(fc="#fce4ec", label="LayerNorm"),
        mpatches.Patch(fc="#fff3e0", label="Output Head"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7.5, framealpha=0.6)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. render_mermaid — Mermaid 流程图
# ---------------------------------------------------------------------------

def render_mermaid(model: nn.Module) -> str:
    """
    生成 Mermaid flowchart 描述模型数据流。
    返回 mermaid 语法字符串，在支持 Mermaid 的 Markdown 里可直接渲染。
    """
    lines = ["```mermaid", "flowchart TD"]

    name = model.__class__.__name__
    lines.append(f"    Model[\\<b\\>{name}\\]:::main")

    # Input
    lines.append("    X[/\"Input x ∈ ℝ^(T×input_dim)\"/]")
    lines.append("    C[/\"Cond c ∈ ℝ^(n_obs×cond_dim)\"/]")
    lines.append("    X --> embx[\"Input\nEmbedding\"]")
    lines.append("    C --> embc[\"Cond\nEmbedding\"]")

    lines.append("    embx --> posx[\"Positional\nEmbedding\"]")
    lines.append("    embc --> posc[\"Cond Pos\nEmbedding\"]")

    if hasattr(model, 'encoder') and model.encoder is not None:
        lines.append("    posc --> enc[\"Condition\nEncoder\"]")
        lines.append("    enc --> mem[\"Memory\"]")

    lines.append("    posx --> dec[\"AttnRes\nDecoder\"]")
    if hasattr(model, 'encoder') and model.encoder is not None:
        lines.append("    mem --> dec")

    lines.append("    dec --> ln[\"Final\nLayerNorm\"]")
    lines.append("    ln --> head[\"Output\nHead\"]")
    lines.append("    head --> out[/\"Output\"/]")

    # 样式
    lines.append("")
    lines.append("    classDef main fill:#1976d2,stroke:#0d47a1,color:#fff")
    lines.append("    classDef data fill:#b3e5fc,stroke:#0277bd,color:#000")
    lines.append("    class X,C,out data")

    lines.append("```")

    mermaid_str = "\n".join(lines)
    print(mermaid_str)
    return mermaid_str

"""core.eval: 评估、可视化指标和训练历史。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn as nn

__all__ = [
    "count_parameters",
    "plot_history",
    "load_wandb_logs",
    "plot_curves",
]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """返回模型的可训练（或全部）参数量。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# plot_history — 来自 trainer.history 的曲线
# ---------------------------------------------------------------------------

def plot_history(
    history: dict,
    keys: list[str] | None = None,
    figsize=(10, 4),
) -> plt.Figure:
    """
    从 DiffusionTrainer.fit() 返回的 history 字典画曲线。

    Parameters
    ----------
    history : dict
        keys: ``train_loss``, ``val_loss``, ``lr``, ``epoch``, ``global_step``
    keys : list of str, optional
        要画的 key，默认 ``['train_loss', 'val_loss']``
    figsize : (float, float)
        matplotlib figure size

    Returns
    -------
    matplotlib.Figure
    """
    if keys is None:
        keys = ["train_loss", "val_loss"]

    # 确定 x 轴：优先 global_step，否则 epoch
    if "global_step" in history and history["global_step"]:
        x_key = "global_step"
    elif "epoch" in history and history["epoch"]:
        x_key = "epoch"
    else:
        x_key = None

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)
    for ax, key in zip(axes.flatten(), keys):
        if key not in history or not history[key]:
            ax.set_axis_off()
            continue
        xs = history.get(x_key, list(range(len(history[key])))) if x_key else list(range(len(history[key])))
        ys = history[key]
        ax.plot(xs, ys, linewidth=1.3, label=key)
        ax.set_title(key)
        ax.set_xlabel(x_key or "step")
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Training History", fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# load_wandb_logs — 读 logs.json.txt（wandb offline 模式写入的）
# ---------------------------------------------------------------------------

def load_wandb_logs(log_dir: str | Path) -> list[dict]:
    """
    读取 Hydra/Wandb offline 模式下写入 ``logs.json.txt`` 的每行 JSON 记录。

    Parameters
    ----------
    log_dir : str | Path
        训练输出目录，含 ``logs.json.txt``

    Returns
    -------
    list[dict]  — 每条记录
    """
    import json

    log_dir = Path(log_dir)
    log_path = log_dir / "logs.json.txt"
    if not log_path.is_file():
        return []

    records: list[dict] = []
    decoder = json.JSONDecoder()
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 找第一个 {
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                obj, _ = decoder.raw_decode(line, brace)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# plot_curves — 多 key 曲线（兼容 wandb logs）
# ---------------------------------------------------------------------------

def plot_curves(
    records: list[dict],
    keys: list[str] | None = None,
    x_key: str = "global_step",
    figsize=(10, 4.5),
    split_sessions: bool = True,
) -> plt.Figure:
    """
    从 load_wandb_logs 读取的 records 列表画曲线。
    自动按 session 切分（跳过 epoch 0 重置），只画最后一段。

    Parameters
    ----------
    records   : list[dict] — load_wandb_logs 的返回值
    keys      : 要画的 metric key 列表，默认 ['train_loss', 'test_mean_score']
    x_key     : x 轴字段，默认 'global_step'
    split_sessions : 是否按训练重启切分并只画最后一段
    figsize   : matplotlib figsize
    """
    if keys is None:
        keys = ["train_loss", "test_mean_score"]

    if not records:
        print("No records to plot.")
        return plt.figure()

    # 切 session
    if split_sessions:
        sessions: list[list[dict]] = []
        cur: list[dict] = []
        prev_epoch = None
        for r in records:
            epoch = r.get("epoch")
            if cur and epoch == 0 and prev_epoch is not None and prev_epoch > 0:
                sessions.append(cur)
                cur = []
            cur.append(r)
            prev_epoch = epoch
        if cur:
            sessions.append(cur)
        session = sessions[-1] if sessions else records
    else:
        session = records

    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)
    for ax, key in zip(axes.flatten(), keys):
        xs, ys = [], []
        for r in session:
            if key in r and r[key] is not None:
                xs.append(r.get(x_key, 0))
                ys.append(r[key])
        if not xs:
            ax.set_axis_off()
            continue
        ax.plot(xs, ys, linewidth=1.3, label=key)
        ax.set_title(key)
        ax.set_xlabel(x_key)
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
        ax.legend()
        best_idx = int(ys.index(max(ys))) if ys else 0
        best_x, best_y = xs[best_idx], ys[best_idx]
        ax.annotate(
            f"best={best_y:.4f}",
            xy=(best_x, best_y),
            xytext=(best_x, best_y * 0.98),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    fig.suptitle("Training Metrics (last session)", fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig

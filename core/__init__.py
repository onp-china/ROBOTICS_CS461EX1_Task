"""mydiffusion/core: 搭积木式实验模块。

从 `core/` 模块导入组件，组合成实验：

- ``core.models`` — 模型变体（AttnRes、baseline）
- ``core.data``  — 数据加载
- ``core.training`` — 训练循环
- ``core.eval`` — 评估、可视化
- ``core.viz``  — 模型架构可视化

Notebook 使用示例::

    from core.models import build_model, count_parameters
    from core.data   import load_mimicgen_lowdim
    from core.training import DiffusionTrainer, TrainingConfig
    from core.eval   import plot_history, load_wandb_logs, plot_curves
    from core.viz    import summary, draw_model, render_mermaid
"""
from __future__ import annotations

__all__ = [
    "models",
    "data",
    "training",
    "eval",
    "viz",
]

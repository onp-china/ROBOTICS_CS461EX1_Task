from typing import Dict, List

import numpy as np
import torch


def compute_batch_metrics(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    success_threshold: float,
) -> Dict[str, float]:
    mse = torch.mean((pred_action - target_action) ** 2).item()
    per_sample_error = torch.mean(torch.abs(pred_action - target_action), dim=(1, 2))
    success_rate = (per_sample_error < success_threshold).float().mean().item()
    cumulative_reward = (-per_sample_error).mean().item()
    return {
        "mse": mse,
        "success_rate": success_rate,
        "cumulative_reward": cumulative_reward,
    }


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: float(np.mean([item[key] for item in metrics_list])) for key in keys}

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.eval.metrics import average_metrics, compute_batch_metrics


@torch.no_grad()
def evaluate_model(model, data_loader: DataLoader, device: torch.device, success_threshold: float) -> Dict:
    model.eval()
    metrics_list: List[Dict[str, float]] = []

    for batch in data_loader:
        obs = batch["obs"].to(device)
        noisy_action = batch["noisy_action"].to(device)
        target_action = batch["target_action"].to(device)
        diffusion_step = batch["diffusion_step"].to(device)

        pred_action, _ = model(obs, noisy_action, diffusion_step)
        metrics_list.append(
            compute_batch_metrics(
                pred_action=pred_action,
                target_action=target_action,
                success_threshold=success_threshold,
            )
        )

    return average_metrics(metrics_list)

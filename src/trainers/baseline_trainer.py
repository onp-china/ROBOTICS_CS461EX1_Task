import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.builder import build_datasets
from src.eval.evaluator import evaluate_model
from src.eval.metrics import compute_batch_metrics
from src.eval.visualize import save_training_curves
from src.models.diffusion_transformer_policy import BaselineDiffusionTransformerPolicy
from src.utils.config import load_config, save_config
from src.utils.device import get_default_device
from src.utils.io import write_history_csv, write_summary_json
from src.utils.seed import set_seed


def build_model(config: Dict) -> BaselineDiffusionTransformerPolicy:
    return BaselineDiffusionTransformerPolicy(
        obs_dim=config["data"]["obs_dim"],
        action_dim=config["data"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        max_diffusion_step=config["model"]["max_diffusion_step"],
        residual_mode=config["model"].get("residual_mode", "standard"),
    )


def run_training(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])

    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, "resolved_config.yaml"))

    device = get_default_device()
    train_dataset, val_dataset = build_datasets(config)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("训练集或验证集为空，请检查 demonstration 数据和切窗配置。")
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    history: List[Dict] = []
    train_start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_metrics: List[Dict] = []

        for step, batch in enumerate(train_loader, start=1):
            try:
                obs = batch["obs"].to(device)
                noisy_action = batch["noisy_action"].to(device)
                target_action = batch["target_action"].to(device)
                diffusion_step = batch["diffusion_step"].to(device)

                pred_action, _ = model(obs, noisy_action, diffusion_step)
                loss = torch.mean((pred_action - target_action) ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["training"]["grad_clip_norm"],
                )
                optimizer.step()

                batch_metrics = compute_batch_metrics(
                    pred_action=pred_action,
                    target_action=target_action,
                    success_threshold=config["training"]["success_threshold"],
                )
                epoch_metrics.append(batch_metrics)

                if step % config["training"]["log_interval"] == 0:
                    print(
                        f"[Epoch {epoch:02d} | Step {step:03d}] "
                        f"loss={loss.item():.6f} "
                        f"success={batch_metrics['success_rate']:.4f}"
                    )
            except RuntimeError as exc:
                print(f"训练时跳过异常 batch: {exc}")
                optimizer.zero_grad(set_to_none=True)
                continue

        train_summary = {
            "train_mse": float(np.mean([m["mse"] for m in epoch_metrics])) if epoch_metrics else float("nan"),
            "train_success_rate": float(np.mean([m["success_rate"] for m in epoch_metrics])) if epoch_metrics else 0.0,
            "train_cumulative_reward": float(np.mean([m["cumulative_reward"] for m in epoch_metrics])) if epoch_metrics else 0.0,
        }

        val_summary = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            success_threshold=config["training"]["success_threshold"],
        )

        row = {
            "epoch": epoch,
            **train_summary,
            "val_mse": val_summary["mse"],
            "val_success_rate": val_summary["success_rate"],
            "val_cumulative_reward": val_summary["cumulative_reward"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d} Done] "
            f"train_mse={row['train_mse']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"val_success={row['val_success_rate']:.4f}"
        )

    total_time = time.time() - train_start_time
    metrics_path = write_history_csv(history, os.path.join(output_dir, "metrics.csv"))
    curve_path = save_training_curves(history, output_dir)
    checkpoint_path = os.path.join(output_dir, "model_final.pt")
    torch.save(model.state_dict(), checkpoint_path)

    summary = {
        "device": str(device),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "epochs": config["training"]["epochs"],
        "total_training_time_sec": total_time,
        "final_metrics": history[-1] if history else {},
        "artifacts": {
            "metrics_csv": metrics_path,
            "training_curve_png": curve_path,
            "checkpoint_pt": checkpoint_path,
        },
        "notes": [
            "当前 synthetic 配置可直接运行。",
            "ManiSkill 正式 baseline 需要替换 demonstration 数据读取和真实评估逻辑。",
        ],
    }
    write_summary_json(summary, os.path.join(output_dir, "summary.json"))
    print(f"full baseline 训练完成，结果已保存到: {output_dir}")

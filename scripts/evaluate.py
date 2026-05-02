import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.builder import build_datasets
from src.eval.evaluator import evaluate_model
from src.models.diffusion_transformer_policy import BaselineDiffusionTransformerPolicy
from src.utils.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 full baseline 评估。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint 路径。")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    _, val_dataset = build_datasets(config)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    device = get_default_device()

    model = BaselineDiffusionTransformerPolicy(
        obs_dim=config["data"]["obs_dim"],
        action_dim=config["data"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        dropout=config["model"]["dropout"],
        max_diffusion_step=config["model"]["max_diffusion_step"],
        residual_mode=config["model"].get("residual_mode", "standard"),
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        success_threshold=config["training"]["success_threshold"],
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

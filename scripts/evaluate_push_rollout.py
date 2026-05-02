import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.mujoco_push_rollout import evaluate_push_policy_rollouts
from src.models.diffusion_transformer_policy import BaselineDiffusionTransformerPolicy
from src.utils.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MuJoCo 两指夹爪 push 任务 rollout 测试。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint 路径。")
    parser.add_argument("--num-rollouts", type=int, default=None, help="覆盖配置里的 rollout 数量。")
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default=None,
        choices=["single_step", "iterative"],
        help="覆盖配置里的采样模式。",
    )
    parser.add_argument(
        "--lock-gripper-open",
        action="store_true",
        help="测试时固定两指夹爪开合，不使用模型预测的 finger 动作。",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
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
    model.eval()

    num_rollouts = args.num_rollouts or int(config.get("evaluation", {}).get("num_rollouts", 20))
    sampler_mode = args.sampler_mode or config.get("evaluation", {}).get("sampler_mode", "single_step")
    lock_gripper_open = bool(args.lock_gripper_open or config.get("evaluation", {}).get("lock_gripper_open", False))
    results = evaluate_push_policy_rollouts(
        model=model,
        config=config,
        device=device,
        num_rollouts=num_rollouts,
        sampler_mode=sampler_mode,
        lock_gripper_open=lock_gripper_open,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

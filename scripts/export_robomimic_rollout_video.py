import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.robomimic_rollout import export_robomimic_rollout_video
from src.models.diffusion_transformer_policy import BaselineDiffusionTransformerPolicy
from src.utils.config import load_config
from src.utils.device import get_default_device
from src.utils.seed import set_seed


def default_output_video(config: dict) -> str:
    output_dir = Path(config["experiment"]["output_dir"])
    return str(output_dir / "robomimic_rollout.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 robomimic low-dim policy 的 rollout 视频。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint 路径。")
    parser.add_argument("--output-video", type=str, default=None, help="输出 mp4 路径。")
    parser.add_argument("--camera-name", type=str, default="frontview", help="robosuite 相机名。")
    parser.add_argument("--width", type=int, default=512, help="输出视频宽度。")
    parser.add_argument("--height", type=int, default=512, help="输出视频高度。")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 rollout 步数上限。")
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default=None,
        choices=["single_step", "iterative"],
        help="覆盖采样模式。",
    )
    parser.add_argument("--robots", type=str, default="Panda", help="当 env_args 缺少 robots 时使用的机器人名。")
    parser.add_argument(
        "--controller-name",
        type=str,
        default="OSC_POSE",
        help="当 env_args 缺少 controller_configs 时尝试加载的 robosuite controller。",
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
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sampler_mode = args.sampler_mode or config.get("evaluation", {}).get("sampler_mode", "single_step")
    summary = export_robomimic_rollout_video(
        model=model,
        config=config,
        device=device,
        output_video=args.output_video or default_output_video(config),
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        sampler_mode=sampler_mode,
        robots=args.robots,
        controller_name=args.controller_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

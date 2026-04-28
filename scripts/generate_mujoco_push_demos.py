import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.mujoco_push_env import build_push_env_from_config
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MuJoCo 两指夹爪 push 任务示教数据。")
    parser.add_argument("--config", required=True, type=str, help="配置文件路径。")
    parser.add_argument("--num-episodes", type=int, default=None, help="覆盖配置里的轨迹数量。")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["data"]["demo_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_push_env_from_config(config, seed=config["experiment"]["seed"])

    demo_cfg = config.get("demo_generation", {})
    num_episodes = args.num_episodes or int(demo_cfg.get("num_episodes", 120))
    minimum_steps = int(demo_cfg.get("minimum_steps", config["data"]["sequence_length"]))
    successes = []
    lengths = []

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=config["experiment"]["seed"] + episode_idx)
        observations = []
        actions = []
        rewards = []
        success = bool(info["success"])

        for _ in range(env.config.episode_steps):
            action = env.expert_action()
            observations.append(obs.astype(np.float32))
            actions.append(action.astype(np.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(np.float32(reward))
            success = success or bool(info["success"])
            if truncated:
                break
            if terminated and len(actions) >= minimum_steps:
                break

        observations_np = np.stack(observations, axis=0).astype(np.float32)
        actions_np = np.stack(actions, axis=0).astype(np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        episode_path = output_dir / f"episode_{episode_idx:04d}.npz"
        np.savez_compressed(
            episode_path,
            observations=observations_np,
            actions=actions_np,
            rewards=rewards_np,
            success=np.asarray(success, dtype=np.bool_),
            goal_xy=info["goal_xy"].astype(np.float32),
            final_puck_xy=info["puck_xy"].astype(np.float32),
            episode_length=np.asarray(len(actions), dtype=np.int32),
        )
        successes.append(1.0 if success else 0.0)
        lengths.append(len(actions))

    summary = {
        "output_dir": str(output_dir),
        "num_episodes": num_episodes,
        "expert_success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

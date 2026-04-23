import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_episode(group: h5py.Group):
    if "actions" not in group:
        raise KeyError(f"{group.name} 缺少 actions 数据集。")

    actions = group["actions"][:].astype(np.float32)

    obs = None
    if "obs" in group:
        obs = group["obs"][:].astype(np.float32)
    elif "observations" in group:
        obs = group["observations"][:].astype(np.float32)

    if obs is None:
        raise KeyError(
            f"{group.name} 没有 obs/observations 数据集。"
            "请先使用 ManiSkill replay_trajectory 把轨迹转换到 obs_mode=state。"
        )

    # ManiSkill 轨迹常见是 obs 长度为 T+1，actions 长度为 T。
    if obs.shape[0] == actions.shape[0] + 1:
        obs = obs[:-1]
    elif obs.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{group.name} 的 obs 与 actions 时间长度不匹配，"
            f"obs.shape={obs.shape}, actions.shape={actions.shape}"
        )

    episode = {
        "observations": obs,
        "actions": actions,
    }
    for optional_key in ["success", "rewards", "reward", "terminated", "truncated", "env_states"]:
        if optional_key in group:
            episode[optional_key] = group[optional_key][:]
    return episode


def pad_and_stack(sequences, pad_value=0.0, dtype=np.float32):
    max_len = max(seq.shape[0] for seq in sequences)
    feature_shape = sequences[0].shape[1:]
    output = np.full((len(sequences), max_len, *feature_shape), pad_value, dtype=dtype)
    lengths = np.zeros((len(sequences),), dtype=np.int32)

    for idx, seq in enumerate(sequences):
        length = seq.shape[0]
        output[idx, :length] = seq
        lengths[idx] = length
    return output, lengths


def convert_h5_to_npz(h5_path: Path, output_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([key for key in f.keys() if key.startswith("traj_")])
        if not traj_keys:
            raise ValueError(f"{h5_path} 中没有找到任何 traj_* 轨迹组。")

        episodes = [extract_episode(f[key]) for key in traj_keys]

    observations, episode_lengths = pad_and_stack([ep["observations"] for ep in episodes], dtype=np.float32)
    actions, _ = pad_and_stack([ep["actions"] for ep in episodes], dtype=np.float32)

    save_dict = {
        "observations": observations,
        "actions": actions,
        "episode_lengths": episode_lengths,
    }

    optional_keys = ["success", "terminated", "truncated", "rewards", "reward"]
    for key in optional_keys:
        values = [ep[key] for ep in episodes if key in ep]
        if len(values) == len(episodes):
            stacked, _ = pad_and_stack(values, dtype=values[0].dtype)
            save_dict[key] = stacked

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    summary = {
        "input_h5": str(h5_path),
        "output_npz": str(output_path),
        "num_episodes": int(observations.shape[0]),
        "max_episode_len": int(observations.shape[1]),
        "obs_dim": int(observations.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "keys": sorted(list(save_dict.keys())),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="把 replay 后的 ManiSkill trajectory.h5 转成 baseline 用的 .npz。")
    parser.add_argument("--input-h5", required=True, type=str, help="输入 trajectory.h5 路径。")
    parser.add_argument("--output-npz", required=True, type=str, help="输出 npz 路径。")
    args = parser.parse_args()

    summary = convert_h5_to_npz(Path(args.input_h5), Path(args.output_npz))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

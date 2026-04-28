from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_episode_major(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} 必须是 [T, D] 或 [E, T, D]，当前 shape={array.shape}")


def _slice_windows(
    observations: np.ndarray,
    actions: np.ndarray,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "observations 和 actions 的 episode 数与时间长度必须一致，"
            f"当前 obs.shape={observations.shape}, actions.shape={actions.shape}"
        )
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数。")
    if stride <= 0:
        raise ValueError("stride 必须为正整数。")

    obs_windows = []
    action_windows = []
    num_episodes, episode_len = observations.shape[:2]

    for episode_idx in range(num_episodes):
        if episode_len < sequence_length:
            continue
        for start in range(0, episode_len - sequence_length + 1, stride):
            end = start + sequence_length
            obs_windows.append(observations[episode_idx, start:end])
            action_windows.append(actions[episode_idx, start:end])

    if not obs_windows:
        raise ValueError(
            f"没有切出任何窗口，请检查 sequence_length={sequence_length} 是否大于轨迹长度。"
        )

    return np.stack(obs_windows, axis=0), np.stack(action_windows, axis=0)


def _load_npz_file(
    file: Path,
    obs_key: str,
    action_key: str,
    sequence_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    data = np.load(file)
    if obs_key not in data or action_key not in data:
        raise KeyError(
            f"{file} 缺少必须字段，当前 keys={list(data.keys())}，"
            f"需要 obs_key={obs_key}, action_key={action_key}"
        )

    raw_obs = data[obs_key].astype(np.float32)
    raw_actions = data[action_key].astype(np.float32)
    observations = _ensure_episode_major(raw_obs, f"{file.name}:{obs_key}")
    actions = _ensure_episode_major(raw_actions, f"{file.name}:{action_key}")
    obs_windows, action_windows = _slice_windows(observations, actions, sequence_length, stride)

    metadata = {
        "file": str(file),
        "raw_obs_shape": list(raw_obs.shape),
        "raw_action_shape": list(raw_actions.shape),
        "windowed_obs_shape": list(obs_windows.shape),
        "windowed_action_shape": list(action_windows.shape),
    }
    return obs_windows, action_windows, metadata


class DemoNPZDataset(Dataset):
    """
    预留给 demonstration npz 文件的基础读取器。

    约定每个 npz 文件至少包含：
    - observations: [N, T, obs_dim]
    - actions: [N, T, action_dim]
    """

    def __init__(
        self,
        files: List[Path],
        obs_key: str,
        action_key: str,
        sequence_length: int,
        stride: int,
        noise_std: float,
        max_diffusion_step: int,
        action_conditioning: str,
        seed: int,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        obs_chunks = []
        action_chunks = []
        self.metadata = []

        for file in files:
            obs_chunk, action_chunk, metadata = _load_npz_file(
                file=file,
                obs_key=obs_key,
                action_key=action_key,
                sequence_length=sequence_length,
                stride=stride,
            )
            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
            self.metadata.append(metadata)

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)

        diffusion_step = rng.integers(
            0,
            max_diffusion_step,
            size=(actions.shape[0],),
            endpoint=False,
        ).astype(np.int64)
        scale = (1.0 + diffusion_step[:, None, None] / max(1, max_diffusion_step - 1)).astype(np.float32)
        noise = rng.normal(size=actions.shape).astype(np.float32) * noise_std * scale
        if action_conditioning == "shifted_history":
            shifted_actions = np.zeros_like(actions, dtype=np.float32)
            shifted_actions[:, 1:, :] = actions[:, :-1, :]
            noisy_action = (shifted_actions + noise).astype(np.float32)
        elif action_conditioning == "zeros":
            noisy_action = np.zeros_like(actions, dtype=np.float32)
        else:
            noisy_action = (actions + noise).astype(np.float32)

        self.obs = torch.from_numpy(observations)
        self.noisy_action = torch.from_numpy(noisy_action)
        self.target_action = torch.from_numpy(actions)
        self.diffusion_step = torch.from_numpy(diffusion_step)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[index],
            "noisy_action": self.noisy_action[index],
            "target_action": self.target_action[index],
            "diffusion_step": self.diffusion_step[index],
        }


def build_demo_npz_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    data_cfg = config["data"]
    demo_root = Path(data_cfg["demo_root"])
    if not demo_root.exists():
        raise FileNotFoundError(f"未找到 demo_root: {demo_root}")

    files = sorted(demo_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"在 {demo_root} 下未找到任何 .npz demonstration 文件。")

    split_ratio = float(data_cfg.get("train_split", 0.9))
    split_idx = max(1, int(len(files) * split_ratio))
    train_files = files[:split_idx]
    val_files = files[split_idx:] or files[:1]

    common_kwargs = {
        "obs_key": data_cfg.get("obs_key", "observations"),
        "action_key": data_cfg.get("action_key", "actions"),
        "sequence_length": data_cfg["sequence_length"],
        "stride": data_cfg.get("stride", data_cfg["sequence_length"]),
        "noise_std": data_cfg["noise_std"],
        "max_diffusion_step": data_cfg["max_diffusion_step"],
        "action_conditioning": data_cfg.get("action_conditioning", "noisy_target"),
    }
    train_dataset = DemoNPZDataset(train_files, seed=config["experiment"]["seed"], **common_kwargs)
    val_dataset = DemoNPZDataset(val_files, seed=config["experiment"]["seed"] + 1, **common_kwargs)
    return train_dataset, val_dataset


def inspect_demo_directory(config: Dict) -> Dict:
    data_cfg = config["data"]
    demo_root = Path(data_cfg["demo_root"])
    if not demo_root.exists():
        raise FileNotFoundError(f"未找到 demo_root: {demo_root}")

    files = sorted(demo_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"在 {demo_root} 下未找到任何 .npz 文件。")

    summaries = []
    total_windows = 0
    for file in files:
        obs_windows, action_windows, metadata = _load_npz_file(
            file=file,
            obs_key=data_cfg.get("obs_key", "observations"),
            action_key=data_cfg.get("action_key", "actions"),
            sequence_length=data_cfg["sequence_length"],
            stride=data_cfg.get("stride", data_cfg["sequence_length"]),
        )
        total_windows += int(obs_windows.shape[0])
        summaries.append(metadata)

    return {
        "demo_root": str(demo_root),
        "num_files": len(files),
        "total_window_count": total_windows,
        "obs_key": data_cfg.get("obs_key", "observations"),
        "action_key": data_cfg.get("action_key", "actions"),
        "sequence_length": data_cfg["sequence_length"],
        "stride": data_cfg.get("stride", data_cfg["sequence_length"]),
        "files": summaries,
    }

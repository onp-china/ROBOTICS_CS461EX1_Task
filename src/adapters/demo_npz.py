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
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            "observations 和 actions 的时间长度必须一致，"
            f"当前 obs.shape={observations.shape}, actions.shape={actions.shape}"
        )
    if sequence_length <= 0:
        raise ValueError("sequence_length 必须为正整数。")
    if stride <= 0:
        raise ValueError("stride 必须为正整数。")

    obs_windows = []
    action_windows = []
    episode_len = observations.shape[0]
    if episode_len >= sequence_length:
        for start in range(0, episode_len - sequence_length + 1, stride):
            end = start + sequence_length
            obs_windows.append(observations[start:end])
            action_windows.append(actions[start:end])

    if not obs_windows:
        raise ValueError(
            f"没有切出任何窗口，请检查 sequence_length={sequence_length} 是否大于轨迹长度。"
        )

    return np.stack(obs_windows, axis=0), np.stack(action_windows, axis=0)


def _extract_episode_lengths(
    data: np.lib.npyio.NpzFile,
    num_episodes: int,
    episode_len: int,
    file: Path,
) -> np.ndarray:
    raw_lengths = data.get("episode_lengths")
    if raw_lengths is None:
        return np.full((num_episodes,), episode_len, dtype=np.int32)

    lengths = np.asarray(raw_lengths).reshape(-1).astype(np.int32)
    if lengths.shape[0] != num_episodes:
        raise ValueError(
            f"{file} 的 episode_lengths 数量与 episode 数不一致，"
            f"num_episodes={num_episodes}, episode_lengths.shape={lengths.shape}"
        )
    if np.any(lengths <= 0):
        raise ValueError(f"{file} 的 episode_lengths 必须全部为正整数，当前={lengths.tolist()}")
    if np.any(lengths > episode_len):
        raise ValueError(
            f"{file} 的 episode_lengths 不能超过 padding 后长度 {episode_len}，"
            f"当前={lengths.tolist()}"
        )
    return lengths


def _load_npz_episodes(
    file: Path,
    obs_key: str,
    action_key: str,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    with np.load(file, allow_pickle=False) as data:
        if obs_key not in data or action_key not in data:
            raise KeyError(
                f"{file} 缺少必须字段，当前 keys={list(data.keys())}，"
                f"需要 obs_key={obs_key}, action_key={action_key}"
            )

        raw_obs = data[obs_key].astype(np.float32)
        raw_actions = data[action_key].astype(np.float32)
        observations = _ensure_episode_major(raw_obs, f"{file.name}:{obs_key}")
        actions = _ensure_episode_major(raw_actions, f"{file.name}:{action_key}")
        if observations.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "observations 和 actions 的 episode 数与时间长度必须一致，"
                f"当前 obs.shape={observations.shape}, actions.shape={actions.shape}"
            )

        num_episodes, padded_episode_len = observations.shape[:2]
        episode_lengths = _extract_episode_lengths(
            data=data,
            num_episodes=num_episodes,
            episode_len=padded_episode_len,
            file=file,
        )

        episodes: List[Tuple[np.ndarray, np.ndarray]] = []
        for episode_idx, valid_length in enumerate(episode_lengths.tolist()):
            obs_episode = observations[episode_idx, :valid_length].copy()
            action_episode = actions[episode_idx, :valid_length].copy()
            episodes.append((obs_episode, action_episode))

        metadata = {
            "file": str(file),
            "num_episodes": int(num_episodes),
            "raw_obs_shape": list(raw_obs.shape),
            "raw_action_shape": list(raw_actions.shape),
            "episode_lengths": episode_lengths.astype(int).tolist(),
        }
        if "source_format" in data:
            metadata["source_format"] = str(np.asarray(data["source_format"]).reshape(()).item())
        if "source_obs_keys" in data:
            metadata["source_obs_keys"] = [str(item) for item in np.asarray(data["source_obs_keys"]).reshape(-1).tolist()]
    return episodes, metadata


class DemoNPZDataset(Dataset):
    """
    预留给 demonstration npz 文件的基础读取器。

    约定每个 npz 文件至少包含：
    - observations: [N, T, obs_dim]
    - actions: [N, T, action_dim]
    """

    def __init__(
        self,
        episodes: List[Tuple[np.ndarray, np.ndarray]],
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

        for episode_idx, (episode_obs, episode_actions) in enumerate(episodes):
            if episode_obs.shape[0] < sequence_length:
                self.metadata.append(
                    {
                        "episode_index": int(episode_idx),
                        "episode_length": int(episode_obs.shape[0]),
                        "window_count": 0,
                        "skipped": True,
                    }
                )
                continue
            obs_chunk, action_chunk = _slice_windows(
                observations=episode_obs,
                actions=episode_actions,
                sequence_length=sequence_length,
                stride=stride,
            )
            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
            self.metadata.append(
                {
                    "episode_index": int(episode_idx),
                    "episode_length": int(episode_obs.shape[0]),
                    "window_count": int(obs_chunk.shape[0]),
                }
            )

        if not obs_chunks:
            raise ValueError(
                f"没有任何 episode 能切出长度为 {sequence_length} 的训练窗口，请检查 demonstration 数据。"
            )

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

    all_episodes: List[Tuple[np.ndarray, np.ndarray]] = []
    for file in files:
        file_episodes, _ = _load_npz_episodes(
            file=file,
            obs_key=data_cfg.get("obs_key", "observations"),
            action_key=data_cfg.get("action_key", "actions"),
        )
        all_episodes.extend(file_episodes)

    if not all_episodes:
        raise ValueError(f"在 {demo_root} 下没有读到任何有效 episode。")

    split_ratio = float(data_cfg.get("train_split", 0.9))
    split_idx = max(1, int(len(all_episodes) * split_ratio))
    train_episodes = all_episodes[:split_idx]
    val_episodes = all_episodes[split_idx:] or all_episodes[:1]

    common_kwargs = {
        "sequence_length": data_cfg["sequence_length"],
        "stride": data_cfg.get("stride", data_cfg["sequence_length"]),
        "noise_std": data_cfg["noise_std"],
        "max_diffusion_step": data_cfg["max_diffusion_step"],
        "action_conditioning": data_cfg.get("action_conditioning", "noisy_target"),
    }
    train_dataset = DemoNPZDataset(train_episodes, seed=config["experiment"]["seed"], **common_kwargs)
    val_dataset = DemoNPZDataset(val_episodes, seed=config["experiment"]["seed"] + 1, **common_kwargs)
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
        episodes, metadata = _load_npz_episodes(
            file=file,
            obs_key=data_cfg.get("obs_key", "observations"),
            action_key=data_cfg.get("action_key", "actions"),
        )
        file_window_count = 0
        for obs_episode, action_episode in episodes:
            if obs_episode.shape[0] < data_cfg["sequence_length"]:
                continue
            obs_windows, _ = _slice_windows(
                observations=obs_episode,
                actions=action_episode,
                sequence_length=data_cfg["sequence_length"],
                stride=data_cfg.get("stride", data_cfg["sequence_length"]),
            )
            file_window_count += int(obs_windows.shape[0])
        total_windows += file_window_count
        metadata["window_count"] = file_window_count
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

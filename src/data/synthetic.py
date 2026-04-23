from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTrajectoryDataset(Dataset):
    """
    教学版 baseline 使用的合成轨迹数据集。
    """

    def __init__(
        self,
        size: int,
        obs_dim: int,
        action_dim: int,
        sequence_length: int,
        noise_std: float,
        max_diffusion_step: int,
        seed: int,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        obs = rng.normal(size=(size, sequence_length, obs_dim)).astype(np.float32)
        mixing = rng.normal(size=(obs_dim, action_dim)).astype(np.float32) / np.sqrt(obs_dim)
        target = np.tanh(obs @ mixing).astype(np.float32)
        diffusion_step = rng.integers(0, max_diffusion_step, size=(size,), endpoint=False).astype(np.int64)

        scale = (1.0 + diffusion_step[:, None, None] / max(1, max_diffusion_step - 1)).astype(np.float32)
        noise = rng.normal(size=target.shape).astype(np.float32) * np.float32(noise_std) * scale
        noisy_action = (target + noise).astype(np.float32)

        self.obs = torch.from_numpy(obs)
        self.noisy_action = torch.from_numpy(noisy_action)
        self.target_action = torch.from_numpy(target)
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

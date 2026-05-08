"""core.data: MimicGen 数据集加载和检查工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

__all__ = ["inspect_dataset", "load_mimicgen_lowdim"]


# ---------------------------------------------------------------------------
# inspect_dataset
# ---------------------------------------------------------------------------

def inspect_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """
    用 h5py 直接读 MimicGen HDF5，探索数据结构，返回摘要。

    Returns
    -------
    dict with keys:
      - n_episodes
      - avg_length, min_length, max_length
      - action_shape : tuple
      - obs_shapes   : dict[key -> tuple]
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    info: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        # ---- demos 下面的 key 是 episode index ----
        demo_keys = [k for k in f.keys() if k.startswith("demo_")]
        n_episodes = len(demo_keys)
        info["n_episodes"] = n_episodes

        lengths = []
        action_shape = None
        obs_shapes: dict[str, tuple] = {}

        for demo_key in sorted(demo_keys, key=lambda x: int(x.split("_")[1]))[:3]:
            grp = f[demo_key]
            # action
            if "action" in grp:
                action_shape = tuple(grp["action"].shape)
            # observations
            if "obs" in grp:
                obs_grp = grp["obs"]
                for obs_key in obs_grp.keys():
                    arr = obs_grp[obs_key][:]
                    shape = tuple(arr.shape)
                    if obs_key not in obs_shapes:
                        obs_shapes[obs_key] = shape
            # episode length from first dataset
            for dset_name in ("action", "obs/robot0_eef_pos"):
                if dset_name in grp or (dset_name == "action" and "action" in grp):
                    lengths.append(grp["action"].shape[0])
                    break

        info["action_shape"] = action_shape or ("<unknown>",)
        info["obs_shapes"] = obs_shapes

        if lengths:
            info["avg_length"] = float(np.mean(lengths))
            info["min_length"] = int(np.min(lengths))
            info["max_length"] = int(np.max(lengths))
        else:
            # 精确重新算一遍
            lengths = []
            for demo_key in demo_keys:
                grp = f[demo_key]
                if "action" in grp:
                    lengths.append(grp["action"].shape[0])
            info["avg_length"] = float(np.mean(lengths)) if lengths else 0
            info["min_length"] = int(np.min(lengths)) if lengths else 0
            info["max_length"] = int(np.max(lengths)) if lengths else 0

    return info


# ---------------------------------------------------------------------------
# load_mimicgen_lowdim
# ---------------------------------------------------------------------------

def load_mimicgen_lowdim(
    dataset_path: str | Path,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    batch_size: int = 256,
    num_workers: int = 4,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    val_ratio: float = 0.02,
    **kwargs,
):
    """
    加载 MimicGen low-dim 绝对动作数据集，返回 (dataset, train_loader, val_loader)。

    Parameters
    ----------
    dataset_path
    horizon, n_obs_steps, n_action_steps
        与 diffusion_policy.dataset.robomimic_replay_lowdim_dataset.RobomimicReplayLowdimDataset 一致。
    batch_size, num_workers, persistent_workers, prefetch_factor
        DataLoader 参数。
    val_ratio
        从训练集划分多少比例给验证集。

    Returns
    -------
    (dataset, train_loader, val_loader)
        dataset 是完整的 RobomimicReplayLowdimDataset；
        train_loader / val_loader 是随机划分后的两个 DataLoader。
    """
    try:
        from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
            RobomimicReplayLowdimDataset,
        )
    except ImportError as exc:
        raise ImportError(
            "load_mimicgen_lowdim 需要 diffusion_policy 包。"
            "请确保 DIFFUSION_POLICY_ROOT 在 sys.path 中。"
        ) from exc

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = RobomimicReplayLowdimDataset(
        dataset_path=str(path),
        horizon=horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        abs_action=True,
        use_legacy_normalizer=False,
        seed=42,
    )

    # 随机划分
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    def _loader(ds, **kw):
        return DataLoader(ds, batch_size=batch_size, **kw)

    train_loader = _loader(
        train_dataset,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = _loader(
        val_dataset,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        shuffle=False,
        pin_memory=True,
    )

    return dataset, train_loader, val_loader

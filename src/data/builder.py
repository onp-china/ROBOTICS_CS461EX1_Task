from typing import Dict, Tuple

from torch.utils.data import Dataset

from src.adapters.demo_npz import build_demo_npz_datasets
from src.adapters.maniskill_stub import build_maniskill_stub
from src.data.synthetic import SyntheticTrajectoryDataset


def build_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    data_cfg = config["data"]
    dataset_type = data_cfg["dataset_type"]

    if dataset_type == "synthetic":
        train_dataset = SyntheticTrajectoryDataset(
            size=data_cfg["train_size"],
            obs_dim=data_cfg["obs_dim"],
            action_dim=data_cfg["action_dim"],
            sequence_length=data_cfg["sequence_length"],
            noise_std=data_cfg["noise_std"],
            max_diffusion_step=data_cfg["max_diffusion_step"],
            seed=config["experiment"]["seed"],
        )
        val_dataset = SyntheticTrajectoryDataset(
            size=data_cfg["val_size"],
            obs_dim=data_cfg["obs_dim"],
            action_dim=data_cfg["action_dim"],
            sequence_length=data_cfg["sequence_length"],
            noise_std=data_cfg["noise_std"],
            max_diffusion_step=data_cfg["max_diffusion_step"],
            seed=config["experiment"]["seed"] + 1,
        )
        return train_dataset, val_dataset

    if dataset_type in {
        "maniskill_demo_npz",
        "demo_npz",
        "mujoco_push_demo_npz",
        "robomimic_demo_npz",
    }:
        return build_demo_npz_datasets(config)

    if dataset_type == "maniskill_stub":
        return build_maniskill_stub(config)

    raise ValueError(f"不支持的 dataset_type: {dataset_type}")

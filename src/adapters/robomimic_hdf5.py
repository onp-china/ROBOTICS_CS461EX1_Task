from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np


def _flatten_timesteps(array: np.ndarray, key: str, demo_key: str) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError(
            f"{demo_key}/obs/{key} 必须至少是二维数组 [T, ...]，当前 shape={array.shape}"
        )
    time_dim = array.shape[0]
    return array.reshape(time_dim, -1).astype(np.float32)


def _get_data_group(file_handle: h5py.File) -> h5py.Group:
    if "data" in file_handle:
        return file_handle["data"]
    return file_handle


def _list_demo_keys(data_group: h5py.Group) -> List[str]:
    demo_keys = [key for key in data_group.keys() if isinstance(data_group[key], h5py.Group)]
    demo_keys = [key for key in demo_keys if key.startswith("demo_")]
    if not demo_keys:
        raise ValueError("没有找到任何 demo_* 轨迹组，请确认输入是 robomimic HDF5 demonstration 文件。")
    return sorted(demo_keys)


def _read_env_args(data_group: h5py.Group) -> str | None:
    env_args = data_group.attrs.get("env_args")
    if env_args is None:
        return None
    if isinstance(env_args, bytes):
        return env_args.decode("utf-8")
    if isinstance(env_args, str):
        return env_args
    return json.dumps(env_args)


def _load_demo(
    data_group: h5py.Group,
    demo_key: str,
    obs_keys: Sequence[str],
    action_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    demo_group = data_group[demo_key]
    if action_key not in demo_group:
        raise KeyError(f"{demo_key} 缺少动作数据集 {action_key}")
    if "obs" not in demo_group:
        raise KeyError(f"{demo_key} 缺少 obs 组。")

    actions = demo_group[action_key][:].astype(np.float32)
    if actions.ndim != 2:
        raise ValueError(f"{demo_key}/{action_key} 必须是 [T, action_dim]，当前 shape={actions.shape}")

    obs_group = demo_group["obs"]
    flattened_obs = []
    expected_steps = actions.shape[0]
    for obs_key in obs_keys:
        if obs_key not in obs_group:
            raise KeyError(
                f"{demo_key}/obs 缺少键 {obs_key}，当前可用键={sorted(list(obs_group.keys()))}"
            )
        obs_array = _flatten_timesteps(obs_group[obs_key][:], key=obs_key, demo_key=demo_key)
        if obs_array.shape[0] != expected_steps:
            raise ValueError(
                f"{demo_key}/obs/{obs_key} 与 actions 时间长度不一致，"
                f"obs.shape={obs_array.shape}, actions.shape={actions.shape}"
            )
        flattened_obs.append(obs_array)

    observations = np.concatenate(flattened_obs, axis=-1)
    return observations, actions


def pad_and_stack(sequences: Sequence[np.ndarray], dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(sequence.shape[0] for sequence in sequences)
    feature_shape = sequences[0].shape[1:]
    output = np.zeros((len(sequences), max_len, *feature_shape), dtype=dtype)
    lengths = np.zeros((len(sequences),), dtype=np.int32)

    for index, sequence in enumerate(sequences):
        length = sequence.shape[0]
        output[index, :length] = sequence
        lengths[index] = length
    return output, lengths


def convert_robomimic_hdf5_to_npz(
    h5_path: Path,
    output_path: Path,
    obs_keys: Sequence[str],
    action_key: str = "actions",
) -> Dict:
    if not obs_keys:
        raise ValueError("obs_keys 不能为空，请至少提供一个 low-dim observation 键。")

    with h5py.File(h5_path, "r") as file_handle:
        data_group = _get_data_group(file_handle)
        demo_keys = _list_demo_keys(data_group)
        env_args = _read_env_args(data_group)
        episodes = [_load_demo(data_group, demo_key, obs_keys=obs_keys, action_key=action_key) for demo_key in demo_keys]

    observation_episodes = [episode_obs for episode_obs, _ in episodes]
    action_episodes = [episode_actions for _, episode_actions in episodes]
    observations, episode_lengths = pad_and_stack(observation_episodes, dtype=np.float32)
    actions, _ = pad_and_stack(action_episodes, dtype=np.float32)

    save_dict = {
        "observations": observations,
        "actions": actions,
        "episode_lengths": episode_lengths,
        "source_format": np.asarray("robomimic_hdf5"),
        "source_obs_keys": np.asarray(list(obs_keys), dtype=np.str_),
        "demo_keys": np.asarray(demo_keys, dtype=np.str_),
    }
    if env_args is not None:
        save_dict["env_args_json"] = np.asarray(env_args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    summary = {
        "input_hdf5": str(h5_path),
        "output_npz": str(output_path),
        "num_episodes": int(observations.shape[0]),
        "max_episode_len": int(observations.shape[1]),
        "obs_dim": int(observations.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "obs_keys": list(obs_keys),
        "action_key": action_key,
        "demo_keys_preview": demo_keys[:5],
    }
    if env_args is not None:
        summary["env_args_json"] = env_args
    return summary

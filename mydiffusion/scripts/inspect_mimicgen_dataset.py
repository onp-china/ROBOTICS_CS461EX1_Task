#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydiffusion._runtime import (
    DEFAULT_LOW_DIM_OBS_KEYS,
    EXPECTED_SINGLE_ARM_RAW_ACTION_DIM,
    MIMICGEN_TASKS,
    add_repo_paths,
    dataset_manifest_path,
    fail,
    infer_max_steps,
    parse_json_maybe,
    processed_dataset_path,
    raw_dataset_path,
    register_mimicgen_environments,
    require_modules,
    to_jsonable,
    write_json,
    write_task_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MimicGen HDF5 datasets and refresh task YAML files.")
    parser.add_argument(
        "--skip-write-task-configs",
        action="store_true",
        help="Only write dataset_manifest.json and keep existing task yaml files untouched.",
    )
    return parser.parse_args()


def feature_dim(shape: tuple[int, ...], key_name: str) -> int:
    if len(shape) != 2:
        fail(
            f"Observation key `{key_name}` is expected to be low-dim with shape [T, D], "
            f"but got shape {shape}."
        )
    return int(shape[-1])


def inspect_task(task_name: str) -> dict:
    import h5py
    from robomimic.utils.file_utils import get_env_metadata_from_dataset

    dataset_path = raw_dataset_path(task_name)
    if not dataset_path.is_file():
        fail(
            f"Missing raw MimicGen dataset for `{task_name}`: {dataset_path}\n"
            "Run `python mydiffusion/scripts/download_mimicgen_datasets.py --mimicgen-repo ...` first."
        )

    env_meta = to_jsonable(get_env_metadata_from_dataset(str(dataset_path)))
    with h5py.File(dataset_path, "r") as handle:
        data_group = handle["data"]
        demo_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
        if not demo_names:
            fail(f"Dataset has no demos: {dataset_path}")

        first_demo = data_group[demo_names[0]]
        obs_group = first_demo["obs"]
        available_obs_keys = sorted(obs_group.keys())
        missing_obs_keys = [key for key in DEFAULT_LOW_DIM_OBS_KEYS if key not in obs_group]
        if missing_obs_keys:
            fail(
                f"Dataset `{dataset_path}` is missing the required low-dim observation keys: "
                f"{missing_obs_keys}. Available keys: {available_obs_keys}"
            )

        obs_shapes = {
            key: list(obs_group[key].shape)
            for key in available_obs_keys
        }
        obs_dim = sum(feature_dim(tuple(obs_group[key].shape), key) for key in DEFAULT_LOW_DIM_OBS_KEYS)

        action_shape = tuple(first_demo["actions"].shape)
        if len(action_shape) != 2:
            fail(f"Expected `actions` to have shape [T, D] in `{dataset_path}`, but got {action_shape}.")
        raw_action_dim = int(action_shape[-1])
        if raw_action_dim != EXPECTED_SINGLE_ARM_RAW_ACTION_DIM:
            fail(
                f"Expected single-arm raw actions to be {EXPECTED_SINGLE_ARM_RAW_ACTION_DIM}D in "
                f"`{dataset_path}`, but found {raw_action_dim}."
            )

        episode_lengths = []
        for demo_name in demo_names:
            demo = data_group[demo_name]
            demo_action_shape = tuple(demo["actions"].shape)
            if len(demo_action_shape) != 2 or demo_action_shape[-1] != raw_action_dim:
                fail(
                    f"Inconsistent action shape in `{dataset_path}` at `{demo_name}`: "
                    f"expected [T, {raw_action_dim}], got {demo_action_shape}."
                )
            episode_lengths.append(int(demo_action_shape[0]))

        env_args = parse_json_maybe(data_group.attrs.get("env_args"))
        max_steps, max_steps_source = infer_max_steps(env_meta, env_args if isinstance(env_args, dict) else None)

    return {
        "task_name": task_name,
        "config_name": f"{task_name}_lowdim_abs",
        "raw_dataset_path": str(dataset_path),
        "processed_dataset_path": str(processed_dataset_path(task_name)),
        "processed_exists": processed_dataset_path(task_name).is_file(),
        "env_meta": env_meta,
        "env_args": to_jsonable(env_args),
        "registered_module": None,
        "obs_shapes": obs_shapes,
        "recommended_obs_keys": list(DEFAULT_LOW_DIM_OBS_KEYS),
        "obs_dim": obs_dim,
        "raw_action_dim": raw_action_dim,
        "episodes": len(episode_lengths),
        "avg_episode_length": float(sum(episode_lengths) / len(episode_lengths)),
        "max_episode_length": max(episode_lengths),
        "min_episode_length": min(episode_lengths),
        "max_steps": max_steps,
        "max_steps_source": max_steps_source,
        "used_fallback_max_steps": max_steps_source == "fallback",
    }


def main() -> None:
    add_repo_paths()
    args = parse_args()
    require_modules(("h5py", "robomimic", "robosuite"))
    registered_module = register_mimicgen_environments()

    tasks = {}
    for task_name in MIMICGEN_TASKS:
        task_payload = inspect_task(task_name)
        task_payload["registered_module"] = registered_module
        tasks[task_name] = task_payload

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_module": registered_module,
        "defaults": {
            "recommended_obs_keys": list(DEFAULT_LOW_DIM_OBS_KEYS),
            "expected_single_arm_raw_action_dim": EXPECTED_SINGLE_ARM_RAW_ACTION_DIM,
            "fallback_max_steps": 1000,
        },
        "tasks": tasks,
    }
    write_json(dataset_manifest_path(), manifest)

    if not args.skip_write_task_configs:
        for task_name, task_payload in tasks.items():
            write_task_config(
                task_name=task_name,
                obs_dim=int(task_payload["obs_dim"]),
                obs_keys=DEFAULT_LOW_DIM_OBS_KEYS,
                max_steps=int(task_payload["max_steps"]),
                manifest_ready=True,
            )

    print(f"Wrote dataset manifest: {dataset_manifest_path()}")
    if args.skip_write_task_configs:
        print("Skipped task yaml refresh as requested.")
    else:
        print("Refreshed task configs with inspected obs_dim / obs_keys / max_steps.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MYDIFFUSION_D1_ROOT = Path(__file__).resolve().parents[1]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import (
    EXPECTED_SINGLE_ARM_RAW_ACTION_DIM,
    MIMICGEN_TASKS,
    add_repo_paths,
    dataset_manifest_path,
    ensure_directory,
    fail,
    processed_dataset_path,
    raw_dataset_path,
    read_json,
    register_mimicgen_environments,
    require_modules,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MimicGen D1 low-dim actions from delta to absolute actions.")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=MIMICGEN_TASKS,
        help="Optional subset of tasks to convert. Defaults to all three tasks.",
    )
    return parser.parse_args()


def validate_raw_dataset(dataset_path: str) -> None:
    import h5py

    with h5py.File(dataset_path, "r") as handle:
        data_group = handle["data"]
        demo_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
        if not demo_names:
            fail(f"Dataset has no demos: {dataset_path}")
        for demo_name in demo_names:
            action_shape = tuple(data_group[demo_name]["actions"].shape)
            if len(action_shape) != 2 or action_shape[-1] != EXPECTED_SINGLE_ARM_RAW_ACTION_DIM:
                fail(
                    f"Expected single-arm raw actions to stay {EXPECTED_SINGLE_ARM_RAW_ACTION_DIM}D in "
                    f"`{dataset_path}`, but `{demo_name}` has shape {action_shape}."
                )


def convert_task(task_name: str) -> None:
    import h5py
    from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

    input_path = raw_dataset_path(task_name)
    output_path = processed_dataset_path(task_name)
    if not input_path.is_file():
        fail(
            f"Missing raw dataset for `{task_name}`: {input_path}\n"
            "Run the downloader and inspection steps first."
        )

    validate_raw_dataset(str(input_path))

    converter = RobomimicAbsoluteActionConverter(str(input_path))
    ensure_directory(output_path.parent)
    shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as output_file:
        for index in range(len(converter)):
            abs_actions = converter.convert_idx(index)
            demo = output_file[f"data/demo_{index}"]
            demo["actions"][:] = abs_actions

    print(f"Converted `{task_name}` to absolute actions: {output_path}")


def update_manifest_processed_flags() -> None:
    manifest_path = dataset_manifest_path()
    if not manifest_path.is_file():
        return
    manifest = read_json(manifest_path)
    tasks = manifest.get("tasks", {})
    changed = False
    for task_name in MIMICGEN_TASKS:
        task_payload = tasks.get(task_name)
        if not isinstance(task_payload, dict):
            continue
        expected = processed_dataset_path(task_name)
        exists = expected.is_file()
        if task_payload.get("processed_dataset_path") != str(expected):
            task_payload["processed_dataset_path"] = str(expected)
            changed = True
        if task_payload.get("processed_exists") != exists:
            task_payload["processed_exists"] = exists
            changed = True
    if changed:
        write_json(manifest_path, manifest)


def main() -> None:
    add_repo_paths()
    args = parse_args()
    require_modules(("h5py", "robomimic", "robosuite"))
    register_mimicgen_environments()

    tasks = args.tasks or list(MIMICGEN_TASKS)
    for task_name in tasks:
        convert_task(task_name)

    update_manifest_processed_flags()


if __name__ == "__main__":
    main()

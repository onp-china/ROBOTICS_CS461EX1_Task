#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
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
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Only diagnose environment replay / controller stages. Do not write processed HDF5 files.",
    )
    parser.add_argument(
        "--demo-idx",
        type=int,
        default=None,
        help="Only diagnose or convert a single demo index.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Only diagnose or convert the first N demos from demo_0.",
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


def resolve_demo_indices(total_demos: int, demo_idx: int | None, max_demos: int | None) -> list[int]:
    if total_demos <= 0:
        return []
    if demo_idx is not None:
        if demo_idx < 0 or demo_idx >= total_demos:
            fail(f"`--demo-idx={demo_idx}` 超出范围，当前数据集 demos={total_demos}。")
        return [demo_idx]
    if max_demos is None:
        return list(range(total_demos))
    if max_demos <= 0:
        fail("`--max-demos` 必须是正整数。")
    return list(range(min(total_demos, max_demos)))


def resolve_output_path(task_name: str, total_demos: int, demo_indices: list[int]) -> tuple[Path, bool]:
    canonical_path = processed_dataset_path(task_name)
    if len(demo_indices) == total_demos and demo_indices == list(range(total_demos)):
        return canonical_path, True
    if len(demo_indices) == 1:
        return canonical_path.parent / f"low_dim_abs.demo_{demo_indices[0]:04d}.hdf5", False
    return canonical_path.parent / f"low_dim_abs.first_{len(demo_indices):04d}.hdf5", False


def print_environment_summary(summary: dict) -> None:
    print(f"[env] task={summary['task_name']} env_name={summary['env_name']} demos={summary['demos']}")
    print(f"[env] dataset={summary['dataset_path']}")
    print(f"[env] env_kwargs_keys={summary['env_kwargs_keys']}")
    print(f"[env] controller={summary['controller_summary']}")


def print_demo_summary(summary: dict, prefix: str) -> None:
    print(
        f"[{prefix}] task={summary['task_name']} demo={summary['demo_idx']} "
        f"steps={summary['num_steps']} action_shape={summary['action_shape']} state_shape={summary['state_shape']}"
    )
    if "obs_keys" in summary:
        print(f"[{prefix}] obs_keys={summary['obs_keys']}")
    print(f"[{prefix}] controller={summary['controller_summary']}")
    if "eval" in summary:
        print(f"[{prefix}] eval={summary['eval']}")


def convert_task(task_name: str, args: argparse.Namespace) -> bool:
    import h5py
    from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

    input_path = raw_dataset_path(task_name)
    if not input_path.is_file():
        fail(
            f"Missing raw dataset for `{task_name}`: {input_path}\n"
            "Run the downloader and inspection steps first."
        )

    validate_raw_dataset(str(input_path))
    converter = RobomimicAbsoluteActionConverter(str(input_path), task_name=task_name)
    canonical_success = False

    try:
        env_summary = converter.describe_environment()
        print_environment_summary(env_summary)

        demo_indices = resolve_demo_indices(len(converter), args.demo_idx, args.max_demos)
        if not demo_indices:
            fail(f"`{task_name}` 没有可诊断 / 可转换的 demos。")

        first_summary = converter.diagnose_demo(demo_indices[0], run_eval=False)
        print_demo_summary(first_summary, prefix="preflight")

        if args.diagnose:
            for demo_idx in demo_indices[1:]:
                summary = converter.diagnose_demo(demo_idx, run_eval=False)
                print_demo_summary(summary, prefix="diagnose")
            print(
                f"[diagnose] `{task_name}` 诊断完成。建议顺序："
                "先 inspect_mimicgen_dataset.py，再用 --diagnose --demo-idx 0，"
                "通过后再扩大到 --max-demos 10，最后再做正式转换。"
            )
            return False

        output_path, is_canonical_output = resolve_output_path(task_name, len(converter), demo_indices)
        ensure_directory(output_path.parent)
        temp_output_path = output_path.with_suffix(output_path.suffix + ".partial")
        if temp_output_path.exists():
            temp_output_path.unlink()

        shutil.copy2(input_path, temp_output_path)
        try:
            with h5py.File(temp_output_path, "r+") as output_file:
                for demo_idx in demo_indices:
                    abs_actions = converter.convert_idx(demo_idx)
                    demo = output_file[f"data/demo_{demo_idx}"]
                    demo["actions"][:] = abs_actions
            os.replace(temp_output_path, output_path)
        except Exception:
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise

        if is_canonical_output:
            print(f"Converted `{task_name}` to absolute actions: {output_path}")
            canonical_success = True
        else:
            print(
                f"Converted subset for `{task_name}`: {output_path}\n"
                "这是局部诊断产物，不会更新 canonical processed dataset 或 manifest。"
            )
        return canonical_success
    finally:
        converter.close()


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
    canonical_success = False
    for task_name in tasks:
        canonical_success = convert_task(task_name, args) or canonical_success

    if not args.diagnose and canonical_success:
        update_manifest_processed_flags()


if __name__ == "__main__":
    main()

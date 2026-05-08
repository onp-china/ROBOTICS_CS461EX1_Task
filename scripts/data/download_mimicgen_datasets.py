#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydiffusion._runtime import (
    MIMICGEN_TASKS,
    add_repo_paths,
    ensure_directory,
    fail,
    find_hdf5_candidates,
    find_mimicgen_repo,
    format_command,
    locate_download_script,
    raw_core_dataset_dir,
    raw_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the three MimicGen D0 datasets via the official downloader.")
    parser.add_argument(
        "--mimicgen-repo",
        default=None,
        help="Path to a local MimicGen checkout. Falls back to MIMICGEN_REPO_ROOT.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the official downloader.",
    )
    return parser.parse_args()


def normalize_download_layout(download_root: Path) -> None:
    target_dir = ensure_directory(raw_core_dataset_dir())
    for task_name in MIMICGEN_TASKS:
        target_path = target_dir / f"{task_name}.hdf5"
        if target_path.is_file():
            continue

        candidates = [
            candidate
            for candidate in find_hdf5_candidates(download_root, task_name)
            if candidate.resolve() != target_path.resolve()
        ]
        if not candidates:
            fail(
                f"Official downloader finished, but `{task_name}.hdf5` was not found under "
                f"{download_root}. Please inspect the MimicGen repo and adjust the wrapper if "
                "the upstream layout changed."
            )
        if len(candidates) > 1:
            fail(
                f"Found multiple downloaded HDF5 candidates for `{task_name}`:\n"
                + "\n".join(f"- {candidate}" for candidate in candidates)
            )
        ensure_directory(target_path.parent)
        shutil.move(str(candidates[0]), str(target_path))

    found = sorted(path.name for path in target_dir.glob("*.hdf5"))
    expected = sorted(f"{task_name}.hdf5" for task_name in MIMICGEN_TASKS)
    if found != expected:
        fail(
            "Expected `mydiffusion/data/mimicgen/raw/core/` to contain exactly the three target "
            f"datasets, but found: {found}"
        )


def stage_existing_targets_for_refresh() -> dict[Path, Path]:
    staged_targets: dict[Path, Path] = {}
    target_dir = ensure_directory(raw_core_dataset_dir())
    for task_name in MIMICGEN_TASKS:
        target_path = target_dir / f"{task_name}.hdf5"
        if not target_path.is_file():
            continue

        backup_path = target_path.with_suffix(target_path.suffix + ".stale")
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(str(target_path), str(backup_path))
        staged_targets[target_path] = backup_path
    return staged_targets


def restore_staged_targets(staged_targets: dict[Path, Path]) -> None:
    for target_path, backup_path in staged_targets.items():
        if target_path.exists():
            target_path.unlink()
        if backup_path.exists():
            ensure_directory(target_path.parent)
            shutil.move(str(backup_path), str(target_path))


def cleanup_staged_targets(staged_targets: dict[Path, Path]) -> None:
    for backup_path in staged_targets.values():
        if backup_path.exists():
            backup_path.unlink()


def main() -> None:
    add_repo_paths()
    args = parse_args()

    repo_root = find_mimicgen_repo(args.mimicgen_repo)
    script_path = locate_download_script(repo_root)
    download_root = ensure_directory(raw_dataset_root())
    staged_targets = stage_existing_targets_for_refresh()
    if staged_targets:
        print(
            "Refreshing existing raw datasets:\n"
            + "\n".join(f"- {target}" for target in staged_targets)
        )

    command = [
        args.python,
        str(script_path),
        "--download_dir",
        str(download_root),
        "--dataset_type",
        "core",
        "--tasks",
        *MIMICGEN_TASKS,
    ]
    print(f"Running official downloader:\n{format_command(command)}")
    try:
        subprocess.run(command, check=True, cwd=repo_root)
        normalize_download_layout(download_root)
    except Exception:
        restore_staged_targets(staged_targets)
        raise
    else:
        cleanup_staged_targets(staged_targets)
    print(f"Downloaded datasets are ready under: {raw_core_dataset_dir()}")


if __name__ == "__main__":
    main()

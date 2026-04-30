#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MYDIFFUSION_D1_ROOT = Path(__file__).resolve().parents[1]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import MIMICGEN_TASKS, MYDIFFUSION_D1_ROOT, add_repo_paths, fail, format_command, task_config_name


TASK_CHOICES = tuple(task_config_name(task_name) for task_name in MIMICGEN_TASKS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MimicGen D1 baseline training job.")
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional training device override, for example `mps`, `cpu`, or `cuda:0`.",
    )
    return parser.parse_args()


def main() -> None:
    add_repo_paths()
    args = parse_args()
    train_script = MYDIFFUSION_D1_ROOT / "train.py"
    if not train_script.is_file():
        fail(f"Training entrypoint is missing: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--config-name=train_diffusion_transformer_mimicgen_d1_lowdim_abs_workspace",
        f"task={args.task}",
        f"training.seed={args.seed}",
    ]
    if args.exp_name:
        command.append(f"exp_name={args.exp_name}")
    if args.device:
        command.append(f"training.device={args.device}")

    print(f"Launching baseline run:\n{format_command(command)}")
    subprocess.run(command, check=True, cwd=MYDIFFUSION_D1_ROOT.parent)


if __name__ == "__main__":
    main()

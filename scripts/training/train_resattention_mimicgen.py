#!/usr/bin/env python3
"""
MimicGen lowdim 任务：ResAttention (AttnRes) Transformer 主干训练入口。

录像（磁盘 mp4 / wandb 视频）只由 env_runner 的 n_train_vis、n_test_vis 决定。
默认（未加 --record-video）会强制二者为 0，不写视频。

training.rollout_every 表示「多久跑一次仿真 rollout 测成功率」，与是否录像无关；
默认改为较大间隔，避免误以为「每 N epoch 在录一段视频」。需要密曲线可显式传小值。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


VALID_TASKS = (
    "three_piece_assembly_d0_lowdim_abs",
)


def default_out_dir(repo_root: Path, task: str, seed: int) -> str:
    slug = task.replace("_lowdim_abs", "").replace("_", "-")
    return str(repo_root / "outputs_resattention" / f"{slug}_seed{seed}")


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    repo_root = Path(__file__).resolve().parents[2]
    train_py = repo_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(train_py)

    parser = argparse.ArgumentParser(description="Train MimicGen task with ResAttention backbone.")
    parser.add_argument(
        "--task",
        required=True,
        choices=VALID_TASKS,
        help="Hydra task 名（与 config/task/*.yaml 一致）。",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Hydra 输出目录；默认 outputs_resattention/<task-short>_seed<seed>",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rollout-every",
        type=int,
        default=50,
        help=(
            "每隔多少个 epoch 做一次仿真 rollout（记录 test/mean_score 等）。"
            "不等于录像；录像见 --record-video。默认 50 与 checkpoint 对齐，减轻仿真开销。"
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--attnres-last-bias-init", type=float, default=8.0)
    parser.add_argument("--attnres-temperature", type=float, default=2.0)
    parser.add_argument("--attnres-blend-init", type=float, default=0.0)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="training.resume=false，不从 checkpoint 续训。",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "开启 rollout 时按 task yaml 录制 mp4（n_train_vis/n_test_vis）；"
            "省略则强制 n_train_vis=0、n_test_vis=0，不生成训练视频。"
        ),
    )
    args = parser.parse_args()

    out_dir = args.out_dir or default_out_dir(repo_root, args.task, args.seed)

    cmd: list[str] = [
        sys.executable,
        str(train_py),
        f"task={args.task}",
        f"training.seed={args.seed}",
        f"training.device={args.device}",
        f"training.rollout_every={args.rollout_every}",
        f"training.checkpoint_every={args.checkpoint_every}",
        "policy.model._target_=resattention_transformer_for_diffusion.ResAttentionTransformerForDiffusion",
        f"+policy.model.attnres_last_bias_init={args.attnres_last_bias_init}",
        f"+policy.model.attnres_temperature={args.attnres_temperature}",
        f"+policy.model.attnres_blend_init={args.attnres_blend_init}",
        f"hydra.run.dir={out_dir}",
        f"hydra.sweep.dir={out_dir}",
        f"multi_run.run_dir={out_dir}",
    ]

    if not args.record_video:
        # 覆盖 task yaml 里的 n_test_vis: 1 等，确保不写 media/*.mp4
        cmd += [
            "task.env_runner.n_train_vis=0",
            "task.env_runner.n_test_vis=0",
        ]

    if args.fresh:
        cmd.append("training.resume=false")

    if args.resume_from:
        if args.fresh:
            parser.error("--fresh and --resume-from are mutually exclusive.")
        ckpt_path = Path(args.resume_from).expanduser().resolve().as_posix()
        cmd.append(f'+training.resume_checkpoint_path="{ckpt_path}"')

    if args.lr is not None:
        cmd.append(f"optimizer.learning_rate={args.lr}")

    print("Launching ResAttention MimicGen training:")
    if not args.record_video:
        print("(录像已关: task.env_runner.n_train_vis=0, n_test_vis=0 — 不在 media/ 写 mp4)")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()

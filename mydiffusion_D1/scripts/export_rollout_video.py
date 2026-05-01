#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

MYDIFFUSION_D1_ROOT = Path(__file__).resolve().parents[1]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import add_repo_paths, fail, register_mimicgen_environments, require_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight rollout video for a trained mydiffusion_D1 task.")
    parser.add_argument("--task", required=True, help="Task config name, e.g. mug_cleanup_d1_lowdim_abs.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, usually checkpoints/latest.ckpt.")
    parser.add_argument("--output-dir", default=None, help="Directory to store rollout artifacts.")
    parser.add_argument("--output-video", default=None, help="Optional explicit mp4 output path.")
    parser.add_argument("--seed", type=int, default=100000, help="Test rollout seed.")
    parser.add_argument("--n-test", type=int, default=1, help="Number of test rollouts to execute.")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs to use.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for inference, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def _resolve_output_dir(checkpoint_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    return run_dir / "rollout_export"


def _collect_media_paths(media_dir: Path) -> set[Path]:
    if not media_dir.is_dir():
        return set()
    return {path.resolve() for path in media_dir.glob("*.mp4")}


def _extract_first_mean_score(log_data: dict) -> float | None:
    for key in ("test/mean_score", "train/mean_score"):
        value = log_data.get(key)
        if value is not None:
            return float(value)
    return None


def main() -> None:
    add_repo_paths()
    args = parse_args()
    require_modules(("hydra", "wandb", "robomimic", "torch", "h5py", "dill", "av", "diffusers", "einops", "scipy", "gym"))
    register_mimicgen_environments()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        fail(f"Checkpoint path does not exist: {checkpoint_path}")

    output_dir = _resolve_output_dir(checkpoint_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import hydra
        import dill
        import torch
    except ModuleNotFoundError as exc:
        fail(f"Unable to import rollout dependency `{exc.name}`. {exc}")

    workspace_cls = hydra.utils.get_class(
        "diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace.TrainDiffusionTransformerLowdimWorkspace"
    )
    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    workspace = workspace_cls(payload["cfg"])
    workspace.load_payload(payload)
    cfg = copy.deepcopy(workspace.cfg)

    task_name = str(getattr(cfg.task, "name", ""))
    if task_name and task_name != args.task:
        fail(
            f"Checkpoint task `{task_name}` does not match requested task `{args.task}`. "
            "Pass a checkpoint from the same task run."
        )

    device = torch.device(args.device)
    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) and workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()

    runner_cfg = copy.deepcopy(cfg.task.env_runner)
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = int(args.n_test)
    runner_cfg.n_test_vis = min(int(args.n_test), 1)
    runner_cfg.test_start_seed = int(args.seed)
    runner_cfg.n_envs = int(args.n_envs)

    media_dir = output_dir / "media"
    before_media = _collect_media_paths(media_dir)
    runner = hydra.utils.instantiate(runner_cfg, output_dir=str(output_dir))
    log_data = runner.run(policy)
    after_media = _collect_media_paths(media_dir)
    new_media = sorted(after_media - before_media)

    exported_video = None
    if new_media:
        exported_video = new_media[0]
        if args.output_video:
            target = Path(args.output_video).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exported_video, target)
            exported_video = target

    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(output_dir),
        "mean_score": _extract_first_mean_score(log_data),
        "video_path": str(exported_video) if exported_video is not None else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

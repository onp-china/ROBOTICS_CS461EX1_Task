#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path

MYDIFFUSION_D1_ROOT = Path(__file__).resolve().parents[1]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import add_repo_paths, fail, read_json_lines, register_mimicgen_environments, require_modules


CHECKPOINT_RE = re.compile(r"epoch=(?P<epoch>\d+)-(?P<metric>[A-Za-z0-9_]+)=(?P<value>-?\d+(?:\.\d+)?)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight rollout video for a trained mydiffusion_D1 task.")
    parser.add_argument("--task", required=True, help="Task config name, e.g. mug_cleanup_d1_lowdim_abs.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path, usually checkpoints/latest.ckpt.")
    parser.add_argument("--run-dir", default=None, help="Run directory used to auto-resolve the best checkpoint.")
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


def _collect_test_rollout_records(log_data: dict) -> list[dict]:
    records = []
    for key, value in log_data.items():
        prefix = "test/sim_video_path_"
        if not key.startswith(prefix):
            continue
        seed = key[len(prefix):]
        record = {
            "seed": int(seed),
            "video_path": str(value),
            "task_success": float(log_data.get(f"test/task_success_{seed}", 0.0)),
            "ready_to_grasp": float(log_data.get(f"test/ready_to_grasp_{seed}", 0.0)),
            "min_eef_object_distance": float(log_data.get(f"test/min_eef_object_distance_{seed}", float("inf"))),
            "object_displacement": float(log_data.get(f"test/object_displacement_{seed}", 0.0)),
        }
        records.append(record)
    records.sort(key=lambda item: item["seed"])
    return records


def _best_rollout_alias_name(record: dict) -> str:
    seed = int(record["seed"])
    if float(record.get("task_success", 0.0)) > 0.0:
        return f"best_success_seed{seed}.mp4"
    return f"best_candidate_seed{seed}.mp4"


def _parse_checkpoint_name(checkpoint_name: str) -> tuple[int, str, float] | None:
    match = CHECKPOINT_RE.match(checkpoint_name)
    if not match:
        return None
    return int(match.group("epoch")), str(match.group("metric")), float(match.group("value"))


def resolve_best_checkpoint_from_run_dir(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        fail(f"Missing checkpoints directory: {checkpoints_dir}")

    log_path = run_dir / "logs.json.txt"
    rows = read_json_lines(log_path)
    if rows:
        rollout_rows = [
            row for row in rows
            if "epoch" in row and any(key in row for key in (
                "test/task_success_rate",
                "test/pregrasp_ready_rate",
                "test/contact_rate",
                "test/mean_min_eef_object_distance",
                "test/mean_object_displacement",
                "test/mean_score",
            ))
        ]
        if rollout_rows:
            best_row = max(
                rollout_rows,
                key=lambda row: (
                    float(row.get("test/task_success_rate", row.get("test/contact_rate", 0.0))),
                    float(row.get("test/pregrasp_ready_rate", 0.0)),
                    -float(row.get("test/mean_min_eef_object_distance", float("inf"))),
                    float(row.get("test/mean_object_displacement", 0.0)),
                    float(row.get("test/mean_score", 0.0)),
                    -float(row.get("val_loss", float("inf"))),
                    int(row.get("epoch", -1)),
                ),
            )
            best_epoch = int(best_row.get("epoch", -1))
            for ckpt_path in sorted(checkpoints_dir.glob(f"epoch={best_epoch:04d}-*.ckpt")):
                return ckpt_path

    candidates = []
    for ckpt_path in sorted(checkpoints_dir.glob("epoch=*-*.ckpt")):
        parsed = _parse_checkpoint_name(ckpt_path.name)
        if parsed is None:
            continue
        epoch, metric_name, metric_value = parsed
        candidates.append((metric_name, metric_value, epoch, ckpt_path))
    if candidates:
        def score(item):
            metric_name, metric_value, epoch, _ = item
            if metric_name == "test_mean_min_eef_object_distance":
                return (3, -metric_value, epoch)
            if metric_name == "test_mean_score":
                return (2, metric_value, epoch)
            if metric_name == "val_loss":
                return (1, -metric_value, epoch)
            return (0, float("-inf"), epoch)
        return max(candidates, key=score)[3]

    latest = checkpoints_dir / "latest.ckpt"
    if latest.is_file():
        return latest
    fail(f"No checkpoint found under: {checkpoints_dir}")


def export_rollout_video_for_checkpoint(
    task: str,
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    output_video: str | Path | None = None,
    seed: int = 100000,
    n_test: int = 1,
    n_envs: int = 1,
    device: str = "cpu",
) -> dict:
    add_repo_paths()
    require_modules(("hydra", "wandb", "robomimic", "torch", "h5py", "dill", "av", "diffusers", "einops", "scipy", "gym"))
    register_mimicgen_environments()

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        fail(f"Checkpoint path does not exist: {checkpoint_path}")

    resolved_output_dir = _resolve_output_dir(
        checkpoint_path=checkpoint_path,
        explicit_output_dir=str(output_dir) if output_dir is not None else None,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

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
    if task_name and task_name != task:
        fail(
            f"Checkpoint task `{task_name}` does not match requested task `{task}`. "
            "Pass a checkpoint from the same task run."
        )

    torch_device = torch.device(device)
    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) and workspace.ema_model is not None else workspace.model
    policy.to(torch_device)
    policy.eval()

    runner_cfg = copy.deepcopy(cfg.task.env_runner)
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = int(n_test)
    runner_cfg.n_test_vis = int(n_test)
    runner_cfg.test_start_seed = int(seed)
    runner_cfg.n_envs = int(n_envs)
    runner_cfg.enable_render = True

    media_dir = resolved_output_dir / "media"
    before_media = _collect_media_paths(media_dir)
    runner = hydra.utils.instantiate(runner_cfg, output_dir=str(resolved_output_dir))
    log_data = runner.run(policy)
    after_media = _collect_media_paths(media_dir)
    new_media = sorted(after_media - before_media)

    rollout_records = _collect_test_rollout_records(log_data)
    exported_video = None
    best_video_path = None
    alias_video_path = None
    if rollout_records:
        best_record = max(
            rollout_records,
            key=lambda item: (
                item["task_success"],
                item["ready_to_grasp"],
                -item["min_eef_object_distance"],
                item["object_displacement"],
                -item["seed"],
            ),
        )
        best_video_path = Path(best_record["video_path"]).expanduser().resolve()
        if best_video_path.is_file():
            alias_video_path = resolved_output_dir / _best_rollout_alias_name(best_record)
            shutil.copy2(best_video_path, alias_video_path)
        if output_video and best_video_path.is_file():
            target = Path(output_video).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_video_path, target)
            exported_video = target
        else:
            exported_video = alias_video_path if alias_video_path is not None else best_video_path
    elif new_media:
        exported_video = new_media[0]
        best_video_path = exported_video
        if output_video:
            target = Path(output_video).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exported_video, target)
            exported_video = target

    return {
        "task": task,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(resolved_output_dir),
        "mean_score": _extract_first_mean_score(log_data),
        "task_success_rate": log_data.get("test/task_success_rate"),
        "pregrasp_ready_rate": log_data.get("test/pregrasp_ready_rate"),
        "best_rollout_video_path": str(best_video_path) if best_video_path is not None else None,
        "best_rollout_alias_path": str(alias_video_path) if alias_video_path is not None else None,
        "rollout_records": rollout_records,
        "video_path": str(exported_video) if exported_video is not None else None,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.run_dir is None:
            fail("Pass either --checkpoint or --run-dir.")
        checkpoint = resolve_best_checkpoint_from_run_dir(args.run_dir)
    summary = export_rollout_video_for_checkpoint(
        task=args.task,
        checkpoint=checkpoint,
        output_dir=args.output_dir,
        output_video=args.output_video,
        seed=args.seed,
        n_test=args.n_test,
        n_envs=args.n_envs,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

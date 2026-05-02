#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mydiffusion._runtime import MYDIFFUSION_ROOT, add_repo_paths, fail, register_mimicgen_environments, require_modules


def validate_training_device(device_name: str) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        fail(f"Unable to validate training device because `{exc.name}` is missing. Install PyTorch first.")

    normalized = str(device_name).strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if has_mps:
            fail(
                f"Configured `training.device={device_name}`, but CUDA is unavailable in this environment. "
                "This machine has Apple Metal support, so rerun with `training.device=mps`. "
                "Example: `python mydiffusion/train.py task=mug_cleanup_d0_lowdim_abs "
                "training.seed=42 training.device=mps`."
            )
        fail(
            f"Configured `training.device={device_name}`, but CUDA is unavailable in this environment. "
            "Rerun with `training.device=cpu`, or move the job to a CUDA-enabled machine."
        )

    if normalized == "mps":
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if not has_mps:
            fail(
                "Configured `training.device=mps`, but PyTorch MPS is unavailable in this environment. "
                "Rerun with `training.device=cpu`, or use a machine where `torch.backends.mps.is_available()` "
                "returns True."
            )


def main() -> None:
    add_repo_paths()
    require_modules(("hydra", "wandb", "robomimic", "robosuite", "torch"))
    register_mimicgen_environments()

    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        fail(f"Unable to import training dependency `{exc.name}`. {exc}")

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path=str(Path(MYDIFFUSION_ROOT, "config")),
        config_name="train_diffusion_transformer_mimicgen_lowdim_abs_workspace",
    )
    def _main(cfg: OmegaConf) -> None:
        OmegaConf.resolve(cfg)
        validate_training_device(str(cfg.training.device))
        manifest_ready = bool(getattr(cfg.task, "manifest_ready", False))
        if not manifest_ready:
            fail(
                "This task config is still a placeholder. Run "
                "`python mydiffusion/scripts/inspect_mimicgen_dataset.py` first so the "
                "real `obs_dim`, `obs_keys`, and `max_steps` are written into the task yaml."
            )

        dataset_path = Path(str(cfg.task.dataset_path)).expanduser()
        if not dataset_path.is_file():
            fail(
                f"Missing converted low-dim absolute-action dataset: {dataset_path}\n"
                "Run `python mydiffusion/scripts/convert_mimicgen_abs_actions.py` after "
                "inspection, then retry the training command."
            )

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.run()

    _main()


if __name__ == "__main__":
    main()

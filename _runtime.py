from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
MYDIFFUSION_ROOT = REPO_ROOT / "mydiffusion"
DIFFUSION_POLICY_REPO_ROOT = REPO_ROOT / "diffusion_policy"

MIMICGEN_REPO_ENV_VAR = "MIMICGEN_REPO_ROOT"
MIMICGEN_TASKS = (
    "three_piece_assembly_d0",
)
MIMICGEN_ENV_NAMES = {
    "three_piece_assembly_d0": "ThreePieceAssembly_D0",
}
DEFAULT_LOW_DIM_OBS_KEYS = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
EXPECTED_SINGLE_ARM_RAW_ACTION_DIM = 7
MODEL_ACTION_DIM = 10
FALLBACK_MAX_STEPS = 1000
PLACEHOLDER_OBS_DIM = -1


def add_repo_paths() -> None:
    for path in (DIFFUSION_POLICY_REPO_ROOT,):
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fail(message: str, exit_code: int = 2) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def module_install_hint(module_name: str) -> str:
    hints = {
        "hydra": "Install Hydra with `pip install hydra-core omegaconf`.",
        "h5py": "Install HDF5 bindings with `pip install h5py`.",
        "wandb": "Install Weights & Biases with `pip install wandb`.",
        "torch": "Install PyTorch with `pip install torch torchvision`.",
        "robomimic": (
            "Install robomimic 0.2.0. On macOS, prefer "
            "`python -m pip install --no-deps robomimic==0.2.0` after installing the "
            "packages listed in `mydiffusion/requirements_mimicgen.txt`."
        ),
        "robosuite": (
            "Install a MimicGen-compatible robosuite version with "
            "`python -m pip install robosuite==1.4.1`. MimicGen does not support "
            "robosuite v1.5+."
        ),
        "mimicgen": (
            "Install MimicGen from a local checkout, for example "
            "`pip install -e /path/to/mimicgen`."
        ),
        "mimicgen_envs": (
            "Install the legacy MimicGen Environments package, for example "
            "`pip install -e /path/to/mimicgen_environments`."
        ),
    }
    return hints.get(module_name, f"Install the missing module `{module_name}`.")


def require_modules(module_names: Sequence[str]) -> None:
    missing = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        detail = "\n".join(
            f"- `{module_name}`: {module_install_hint(module_name)}"
            for module_name in missing
        )
        fail(
            "Missing required Python modules:\n"
            f"{detail}\n"
            "Also review `mydiffusion/requirements_mimicgen.txt` for the tested package set."
        )


def register_mimicgen_environments() -> str:
    for module_name in ("mimicgen", "mimicgen_envs"):
        try:
            importlib.import_module(module_name)
            registered_env_names = _collect_registered_env_names()
            if not registered_env_names:
                fail(
                    f"Imported `{module_name}`, but could not inspect any robosuite environment registry entries. "
                    "This usually means the robosuite / MimicGen installation is incomplete or version-mismatched. "
                    "Make sure `robosuite`, `robomimic`, and the MimicGen package are installed in the same environment."
                )

            expected_env_names = set(MIMICGEN_ENV_NAMES.values())
            missing_env_names = sorted(expected_env_names - registered_env_names)
            if missing_env_names:
                sample_env_names = ", ".join(sorted(registered_env_names)[:10])
                fail(
                    f"Imported `{module_name}`, but the required MimicGen tasks are not registered in robosuite. "
                    f"Missing env names: {', '.join(missing_env_names)}. "
                    f"Registry sample: {sample_env_names or '<empty>'}. "
                    "This usually means MimicGen imported without fully registering its robosuite environments. "
                    "Reinstall the matching `robosuite` / `robomimic` / MimicGen versions and verify that "
                    "`import mimicgen` or `import mimicgen_envs` registers these tasks."
                )

            return module_name
        except ModuleNotFoundError:
            continue
    fail(
        "Unable to register MimicGen environments. Install either the current `mimicgen` "
        "package or the legacy `mimicgen_envs` package before running this command."
    )


def _coerce_env_name_iterable(value: Any) -> set[str]:
    if value is None:
        return set()

    iterable: Any
    if isinstance(value, Mapping):
        iterable = value.keys()
    elif isinstance(value, (set, list, tuple)):
        iterable = value
    else:
        try:
            iterable = iter(value)
        except TypeError:
            return set()

    result = set()
    for item in iterable:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            result.add(item)
    return result


def _collect_registered_env_names() -> set[str]:
    registered = set()

    try:
        import robosuite

        registered.update(_coerce_env_name_iterable(getattr(robosuite, "ALL_ENVIRONMENTS", None)))
        registered.update(_coerce_env_name_iterable(getattr(robosuite, "REGISTERED_ENVS", None)))
    except Exception:
        pass

    try:
        from robosuite.environments import ALL_ENVIRONMENTS

        registered.update(_coerce_env_name_iterable(ALL_ENVIRONMENTS))
    except Exception:
        pass

    try:
        from robosuite.environments.base import REGISTERED_ENVS

        registered.update(_coerce_env_name_iterable(REGISTERED_ENVS))
    except Exception:
        pass

    return registered


def find_mimicgen_repo(explicit_repo: str | None) -> Path:
    repo_hint = explicit_repo or os.environ.get(MIMICGEN_REPO_ENV_VAR)
    if not repo_hint:
        fail(
            "No MimicGen repository was provided. Pass `--mimicgen-repo /path/to/mimicgen` "
            f"or export `{MIMICGEN_REPO_ENV_VAR}`."
        )
    repo_root = Path(repo_hint).expanduser().resolve()
    if not repo_root.exists():
        fail(f"MimicGen repository path does not exist: {repo_root}")
    return repo_root


def locate_download_script(repo_root: Path) -> Path:
    candidates = (
        repo_root / "mimicgen" / "scripts" / "download_datasets.py",
        repo_root / "mimicgen_envs" / "scripts" / "download_datasets.py",
        repo_root / "scripts" / "download_datasets.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fail(
        "Could not find the official MimicGen dataset downloader under the provided repo. "
        "Expected one of:\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
    )


def raw_dataset_root() -> Path:
    return MYDIFFUSION_ROOT / "data" / "mimicgen" / "raw"


def raw_core_dataset_dir() -> Path:
    return raw_dataset_root() / "core"


def raw_dataset_path(task_name: str) -> Path:
    return raw_core_dataset_dir() / f"{task_name}.hdf5"


def processed_dataset_path(task_name: str) -> Path:
    return MYDIFFUSION_ROOT / "data" / "mimicgen" / "processed" / task_name / "low_dim_abs.hdf5"


def reports_dir() -> Path:
    return MYDIFFUSION_ROOT / "reports"


def outputs_dir() -> Path:
    return MYDIFFUSION_ROOT / "outputs"


def task_config_name(task_name: str) -> str:
    return f"{task_name}_lowdim_abs"


def task_config_path(task_name: str) -> Path:
    return MYDIFFUSION_ROOT / "config" / "task" / f"{task_config_name(task_name)}.yaml"


def dataset_manifest_path() -> Path:
    return reports_dir() / "dataset_manifest.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def parse_json_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return to_jsonable(value)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def infer_max_steps(env_meta: Mapping[str, Any] | None, env_args: Mapping[str, Any] | None) -> tuple[int, str]:
    candidates = (
        ("env_meta.env_kwargs.horizon", nested_get(env_meta, ("env_kwargs", "horizon"))),
        ("env_meta.env_kwargs.max_steps", nested_get(env_meta, ("env_kwargs", "max_steps"))),
        ("env_args.env_kwargs.horizon", nested_get(env_args, ("env_kwargs", "horizon"))),
        ("env_args.env_kwargs.max_steps", nested_get(env_args, ("env_kwargs", "max_steps"))),
        ("env_args.horizon", nested_get(env_args, ("horizon",))),
        ("env_args.max_steps", nested_get(env_args, ("max_steps",))),
    )
    for source, value in candidates:
        if isinstance(value, int) and value > 0:
            return value, source
    return FALLBACK_MAX_STEPS, "fallback"


def nested_get(mapping: Mapping[str, Any] | None, keys: Sequence[str]) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def task_config_yaml(
    task_name: str,
    obs_dim: int,
    obs_keys: Sequence[str],
    max_steps: int,
    manifest_ready: bool,
) -> str:
    config_name = task_config_name(task_name)
    dataset_path = processed_dataset_path(task_name).as_posix()
    obs_keys_literal = "[" + ", ".join(repr(key) for key in obs_keys) + "]"
    return f"""# Auto-generated by mydiffusion/scripts/inspect_mimicgen_dataset.py
name: {config_name}
manifest_ready: {str(manifest_ready)}

obs_dim: {obs_dim}
action_dim: {MODEL_ACTION_DIM}
keypoint_dim: 3

obs_keys: &obs_keys {obs_keys_literal}
task_name: &task_name {task_name}
abs_action: &abs_action True
dataset_path: &dataset_path {dataset_path}

env_runner:
  _target_: diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner
  dataset_path: *dataset_path
  obs_keys: *obs_keys
  n_train: 2
  n_train_vis: 0
  train_start_idx: 0
  n_test: 20
  n_test_vis: 2
  test_start_seed: 100000
  max_steps: {max_steps}
  n_obs_steps: ${{n_obs_steps}}
  n_action_steps: ${{n_action_steps}}
  n_latency_steps: ${{n_latency_steps}}
  render_hw: [128, 128]
  fps: 10
  crf: 22
  past_action: ${{past_action_visible}}
  abs_action: *abs_action
  n_envs: 8

dataset:
  _target_: diffusion_policy.dataset.robomimic_replay_lowdim_dataset.RobomimicReplayLowdimDataset
  dataset_path: *dataset_path
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1+${{n_latency_steps}}'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  obs_keys: *obs_keys
  abs_action: *abs_action
  use_legacy_normalizer: False
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null
"""


def write_task_config(
    task_name: str,
    obs_dim: int,
    obs_keys: Sequence[str],
    max_steps: int,
    manifest_ready: bool,
) -> Path:
    path = task_config_path(task_name)
    ensure_directory(path.parent)
    path.write_text(
        task_config_yaml(
            task_name=task_name,
            obs_dim=obs_dim,
            obs_keys=obs_keys,
            max_steps=max_steps,
            manifest_ready=manifest_ready,
        ),
        encoding="utf-8",
    )
    return path


def write_placeholder_task_configs() -> None:
    for task_name in MIMICGEN_TASKS:
        write_task_config(
            task_name=task_name,
            obs_dim=PLACEHOLDER_OBS_DIM,
            obs_keys=DEFAULT_LOW_DIM_OBS_KEYS,
            max_steps=FALLBACK_MAX_STEPS,
            manifest_ready=False,
        )


def find_hdf5_candidates(root: Path, task_name: str) -> list[Path]:
    exact = []
    fuzzy = []
    for candidate in root.rglob("*.hdf5"):
        if candidate.name == f"{task_name}.hdf5":
            exact.append(candidate)
        elif task_name in candidate.stem:
            fuzzy.append(candidate)
    return exact or fuzzy


def format_command(argv: Sequence[str]) -> str:
    return " ".join(argv)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def population_std(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    avg = mean(numbers)
    variance = sum((value - avg) ** 2 for value in numbers) / len(numbers)
    return variance ** 0.5

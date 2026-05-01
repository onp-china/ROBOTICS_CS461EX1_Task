from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


OBS_KEY_ALIASES = {
    "object": ["object-state"],
    "object-state": ["object"],
}


def build_observation_window(observations: List[np.ndarray], sequence_length: int) -> np.ndarray:
    if not observations:
        raise ValueError("observations 不能为空。")

    if len(observations) >= sequence_length:
        window = observations[-sequence_length:]
    else:
        pad_count = sequence_length - len(observations)
        window = [observations[0]] * pad_count + observations
    return np.stack(window, axis=0).astype(np.float32)


def build_action_window(actions: List[np.ndarray], sequence_length: int, action_dim: int) -> np.ndarray:
    zero_action = np.zeros((action_dim,), dtype=np.float32)
    if len(actions) >= sequence_length:
        history = actions[-sequence_length:]
    else:
        history = [zero_action] * (sequence_length - len(actions)) + actions
    return np.stack(history, axis=0).astype(np.float32)


@torch.no_grad()
def sample_action_sequence(
    model,
    obs_window: torch.Tensor,
    action_dim: int,
    max_diffusion_step: int,
    sampler_mode: str = "single_step",
    initial_action_window: torch.Tensor | None = None,
    single_step_diffusion_step: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length, _ = obs_window.shape
    if initial_action_window is None:
        initial_action_window = torch.zeros(
            (batch_size, sequence_length, action_dim),
            dtype=obs_window.dtype,
            device=obs_window.device,
        )

    if sampler_mode == "iterative":
        action_estimate = initial_action_window.clone()
        for step in reversed(range(max_diffusion_step)):
            diffusion_step = torch.full((batch_size,), step, dtype=torch.long, device=obs_window.device)
            action_estimate, _ = model(obs_window, action_estimate, diffusion_step)
        return action_estimate

    diffusion_step = torch.full(
        (batch_size,),
        int(single_step_diffusion_step),
        dtype=torch.long,
        device=obs_window.device,
    )
    action_estimate, _ = model(obs_window, initial_action_window, diffusion_step)
    return action_estimate


def _load_required_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"缺少依赖 {module_name}。请先安装：{install_hint}"
        ) from exc


def _parse_env_args(env_args_json: str | None) -> Dict:
    if not env_args_json:
        return {}
    try:
        return json.loads(env_args_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"env_args_json 不是合法 JSON: {env_args_json}") from exc


def _load_demo_metadata(config: Dict) -> Dict:
    demo_root = Path(config["data"]["demo_root"])
    files = sorted(demo_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"在 {demo_root} 下未找到任何 .npz demonstration 文件。")

    first_file = files[0]
    metadata = {
        "npz_path": str(first_file),
        "env_args": {},
        "source_obs_keys": list(config["data"].get("source_obs_keys", [])),
    }
    with np.load(first_file, allow_pickle=False) as data:
        if not metadata["source_obs_keys"] and "source_obs_keys" in data:
            metadata["source_obs_keys"] = [str(item) for item in np.asarray(data["source_obs_keys"]).reshape(-1).tolist()]
        if "env_args_json" in data:
            metadata["env_args"] = _parse_env_args(str(np.asarray(data["env_args_json"]).reshape(()).item()))

    if not metadata["source_obs_keys"]:
        raise ValueError(
            f"{first_file} 和当前配置里都没有 source_obs_keys，无法把 low-dim obs 拼成模型输入。"
        )
    return metadata


def _load_controller_config(controller_name: str):
    try:
        import robosuite as suite  # type: ignore

        if hasattr(suite, "load_controller_config"):
            try:
                return suite.load_controller_config(default_controller=controller_name)
            except TypeError:
                return suite.load_controller_config(controller=controller_name)
    except Exception:
        pass

    try:
        from robosuite.controllers import load_controller_config  # type: ignore

        return load_controller_config(default_controller=controller_name)
    except Exception:
        return None


def _extract_unexpected_kwarg(exc: TypeError) -> str | None:
    match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", str(exc))
    if match is None:
        return None
    return str(match.group(1))


def _make_robosuite_env(
    env_name: str,
    env_kwargs: Dict,
    robots: str,
    camera_name: str,
    controller_name: str,
    horizon: int | None,
) -> object:
    suite = _load_required_dependency("robosuite", "pip install robosuite")

    rollout_kwargs = dict(env_kwargs)
    metadata_only_keys = {
        "env_name",
        "camera_names",
        "camera_heights",
        "camera_widths",
        "camera_depths",
        "camera_segmentations",
    }
    for key in metadata_only_keys:
        rollout_kwargs.pop(key, None)
    original_metadata_keys = set(rollout_kwargs.keys())
    rollout_kwargs["robots"] = rollout_kwargs.get("robots", robots)
    rollout_kwargs["has_renderer"] = False
    rollout_kwargs["has_offscreen_renderer"] = True
    rollout_kwargs["use_camera_obs"] = False
    rollout_kwargs["ignore_done"] = False
    if horizon is not None:
        rollout_kwargs["horizon"] = horizon

    controller_configs = rollout_kwargs.get("controller_configs")
    if controller_configs is None:
        controller_configs = _load_controller_config(controller_name)
        if controller_configs is not None:
            rollout_kwargs["controller_configs"] = controller_configs

    dropped_kwargs: List[str] = []
    while True:
        try:
            return suite.make(env_name=env_name, **rollout_kwargs)
        except TypeError as exc:
            unsupported_key = _extract_unexpected_kwarg(exc)
            if unsupported_key is None or unsupported_key not in original_metadata_keys or unsupported_key not in rollout_kwargs:
                raise TypeError(
                    f"创建 robosuite 环境失败。env_name={env_name}, "
                    f"kwargs={sorted(rollout_kwargs.keys())}, dropped={sorted(dropped_kwargs)}"
                ) from exc
            rollout_kwargs.pop(unsupported_key, None)
            dropped_kwargs.append(unsupported_key)


def _resolve_action_conditioning(config: Dict) -> str:
    return str(config.get("data", {}).get("action_conditioning", "noisy_target"))


def _build_initial_action_window(
    action_history: List[np.ndarray],
    sequence_length: int,
    action_dim: int,
    action_conditioning: str,
    noise_std: float,
    max_diffusion_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if action_conditioning == "shifted_history":
        return build_action_window(action_history, sequence_length, action_dim)
    if action_conditioning == "zeros":
        return np.zeros((sequence_length, action_dim), dtype=np.float32)
    if action_conditioning == "noisy_target":
        max_scale = 1.0 + (max_diffusion_step - 1) / max(1, max_diffusion_step - 1)
        return rng.normal(
            loc=0.0,
            scale=noise_std * max_scale,
            size=(sequence_length, action_dim),
        ).astype(np.float32)
    raise ValueError(f"不支持的 action_conditioning: {action_conditioning}")


def _resolve_single_step_diffusion_step(action_conditioning: str, max_diffusion_step: int) -> int:
    if action_conditioning == "noisy_target":
        return max(0, max_diffusion_step - 1)
    return 0


def _flatten_low_dim_obs(obs: Dict, obs_keys: Sequence[str]) -> np.ndarray:
    chunks = []
    missing_keys = []
    for key in obs_keys:
        resolved_key = key
        if resolved_key not in obs:
            for alias in OBS_KEY_ALIASES.get(key, []):
                if alias in obs:
                    resolved_key = alias
                    break
        if resolved_key not in obs:
            missing_keys.append(key)
            continue
        value = np.asarray(obs[resolved_key], dtype=np.float32).reshape(-1)
        chunks.append(value)

    if missing_keys:
        raise KeyError(
            f"当前环境返回的 obs 缺少键 {missing_keys}，可用键包括 {sorted(list(obs.keys()))}"
        )
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _extract_reset_obs(reset_result) -> Dict:
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _extract_step_result(step_result) -> Tuple[Dict, float, bool, bool, Dict]:
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, float(reward), bool(done), False, info
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, float(reward), bool(terminated), bool(truncated), info
    raise ValueError(f"不支持的 env.step 返回长度: {len(step_result)}")


def _extract_success(env, info: Dict) -> bool:
    success = False
    for key in ["success", "is_success", "task_success", "Success"]:
        if key in info:
            success = success or bool(np.asarray(info[key]).reshape(()).item())

    if hasattr(env, "_check_success"):
        try:
            success = success or bool(env._check_success())
        except Exception:
            pass

    if hasattr(env, "is_success"):
        try:
            result = env.is_success()
            if isinstance(result, dict):
                success = success or any(bool(value) for value in result.values())
            else:
                success = success or bool(result)
        except Exception:
            pass

    return success


def _render_frame(env, camera_name: str, width: int, height: int) -> np.ndarray:
    if not hasattr(env, "sim") or not hasattr(env.sim, "render"):
        raise AttributeError("当前 robosuite 环境没有可用的 sim.render 接口。")
    frame = env.sim.render(width=width, height=height, camera_name=camera_name)
    return np.flipud(np.asarray(frame, dtype=np.uint8))


def _get_action_bounds(env) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(env, "action_spec"):
        return None, None
    try:
        low, high = env.action_spec
    except Exception:
        return None, None
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


@torch.no_grad()
def export_robomimic_rollout_video(
    model,
    config: Dict,
    device: torch.device,
    output_video: str,
    camera_name: str = "frontview",
    width: int = 512,
    height: int = 512,
    max_steps: int | None = None,
    sampler_mode: str = "single_step",
    robots: str = "Panda",
    controller_name: str = "OSC_POSE",
) -> Dict:
    imageio = _load_required_dependency(
        "imageio",
        "pip install imageio imageio-ffmpeg",
    )
    metadata = _load_demo_metadata(config)
    env_args = metadata["env_args"]
    env_name = str(env_args.get("env_name") or config["data"].get("task_name") or "Lift")
    env_kwargs = dict(env_args.get("env_kwargs", {}))

    env = _make_robosuite_env(
        env_name=env_name,
        env_kwargs=env_kwargs,
        robots=robots,
        camera_name=camera_name,
        controller_name=controller_name,
        horizon=max_steps,
    )

    sequence_length = int(config["data"]["sequence_length"])
    action_dim = int(config["data"]["action_dim"])
    max_diffusion_step = int(config["model"]["max_diffusion_step"])
    noise_std = float(config["data"]["noise_std"])
    action_conditioning = _resolve_action_conditioning(config)
    obs_keys = metadata["source_obs_keys"]
    action_low, action_high = _get_action_bounds(env)
    if action_low is not None and action_low.size != action_dim:
        raise ValueError(
            f"配置里的 action_dim={action_dim} 与 robosuite 环境动作维度 {action_low.size} 不一致。"
        )

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=max(1, int(round(getattr(env, "control_freq", 20)))))

    try:
        reset_obs = _extract_reset_obs(env.reset())
        initial_obs = _flatten_low_dim_obs(reset_obs, obs_keys)
        if initial_obs.shape[0] != int(config["data"]["obs_dim"]):
            raise ValueError(
                f"配置里的 obs_dim={config['data']['obs_dim']} 与 rollout 环境拼接后的维度 {initial_obs.shape[0]} 不一致。"
            )
        obs_history = [initial_obs]
        action_history: List[np.ndarray] = []
        total_reward = 0.0
        success = False
        frames = 0
        step_count = 0
        horizon = int(max_steps or getattr(env, "horizon", 200))
        rng = np.random.default_rng(int(config["experiment"]["seed"]))

        writer.append_data(_render_frame(env, camera_name=camera_name, width=width, height=height))
        frames += 1

        while step_count < horizon:
            obs_window_np = build_observation_window(obs_history, sequence_length)
            action_window_np = _build_initial_action_window(
                action_history=action_history,
                sequence_length=sequence_length,
                action_dim=action_dim,
                action_conditioning=action_conditioning,
                noise_std=noise_std,
                max_diffusion_step=max_diffusion_step,
                rng=rng,
            )
            obs_window = torch.from_numpy(obs_window_np).unsqueeze(0).to(device)
            action_window = torch.from_numpy(action_window_np).unsqueeze(0).to(device)
            action_sequence = sample_action_sequence(
                model=model,
                obs_window=obs_window,
                action_dim=action_dim,
                max_diffusion_step=max_diffusion_step,
                sampler_mode=sampler_mode,
                initial_action_window=action_window,
                single_step_diffusion_step=_resolve_single_step_diffusion_step(
                    action_conditioning=action_conditioning,
                    max_diffusion_step=max_diffusion_step,
                ),
            )
            action = action_sequence[0, -1].detach().cpu().numpy().astype(np.float32)
            if action_low is not None and action_high is not None:
                action = np.clip(action, action_low, action_high)
            next_obs, reward, terminated, truncated, info = _extract_step_result(env.step(action))
            obs_history.append(_flatten_low_dim_obs(next_obs, obs_keys))
            action_history.append(action)
            total_reward += reward
            step_count += 1
            success = success or _extract_success(env, info)
            writer.append_data(_render_frame(env, camera_name=camera_name, width=width, height=height))
            frames += 1
            if terminated or truncated:
                break
    finally:
        writer.close()
        if hasattr(env, "close"):
            env.close()

    return {
        "video_path": str(output_path),
        "env_name": env_name,
        "obs_keys": list(obs_keys),
        "steps": step_count,
        "frames": frames,
        "success": bool(success),
        "total_reward": float(total_reward),
        "camera_name": camera_name,
        "width": int(width),
        "height": int(height),
        "sampler_mode": sampler_mode,
    }

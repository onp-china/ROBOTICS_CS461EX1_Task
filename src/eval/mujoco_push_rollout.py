from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.envs.mujoco_push_env import build_push_env_from_config


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

    diffusion_step = torch.zeros((batch_size,), dtype=torch.long, device=obs_window.device)
    action_estimate, _ = model(obs_window, initial_action_window, diffusion_step)
    return action_estimate


@torch.no_grad()
def evaluate_push_policy_rollouts(
    model,
    config: Dict,
    device: torch.device,
    num_rollouts: int,
    sampler_mode: str = "single_step",
    lock_gripper_open: bool = False,
) -> Dict:
    env = build_push_env_from_config(config, seed=config["experiment"]["seed"])
    sequence_length = config["data"]["sequence_length"]
    action_dim = config["data"]["action_dim"]
    max_diffusion_step = config["model"]["max_diffusion_step"]

    returns: List[float] = []
    final_distances: List[float] = []
    success_flags: List[float] = []
    episode_lengths: List[int] = []

    for rollout_idx in range(num_rollouts):
        obs, info = env.reset(seed=config["experiment"]["seed"] + rollout_idx)
        obs_history = [obs]
        action_history: List[np.ndarray] = []
        total_reward = 0.0
        success = bool(info["success"])
        steps = 0

        while steps < env.config.episode_steps:
            obs_window_np = build_observation_window(obs_history, sequence_length)
            action_window_np = build_action_window(action_history, sequence_length, action_dim)
            obs_window = torch.from_numpy(obs_window_np).unsqueeze(0).to(device)
            action_window = torch.from_numpy(action_window_np).unsqueeze(0).to(device)
            action_sequence = sample_action_sequence(
                model=model,
                obs_window=obs_window,
                action_dim=action_dim,
                max_diffusion_step=max_diffusion_step,
                sampler_mode=sampler_mode,
                initial_action_window=action_window,
            )
            action = action_sequence[0, -1].detach().cpu().numpy()
            if lock_gripper_open and action.shape[0] >= 4:
                action[2:] = np.float32(config.get("environment", {}).get("finger_open", 0.055))

            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            action_history.append(action.astype(np.float32))
            total_reward += reward
            success = success or bool(info["success"])
            steps += 1
            if terminated or truncated:
                break

        returns.append(float(total_reward))
        final_distances.append(float(info["box_goal_distance"]))
        success_flags.append(1.0 if success else 0.0)
        episode_lengths.append(steps)

    return {
        "num_rollouts": num_rollouts,
        "sampler_mode": sampler_mode,
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }

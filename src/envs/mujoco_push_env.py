from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np


MUJOCO_PUSH_XML = """
<mujoco model="two_finger_push">
  <compiler angle="radian"/>
  <option gravity="0 0 0" integrator="implicitfast" timestep="0.005"/>

  <default>
    <joint damping="1.0" limited="true"/>
    <geom condim="3" friction="0.8 0.01 0.0005" rgba="0.7 0.7 0.7 1"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.05" size="1 1 0.05" rgba="0.95 0.95 0.95 1"/>
    <site name="goal" type="cylinder" pos="0.75 0 0" size="0.055 0.003" rgba="0.2 0.8 0.2 0.45"/>

    <body name="arm_base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.03" rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-1.4 1.4"/>
        <geom type="capsule" fromto="0 0 0 0.45 0 0" size="0.032" rgba="0.3 0.5 0.8 1" contype="0" conaffinity="0"/>

        <body name="link2" pos="0.45 0 0">
          <joint name="elbow" type="hinge" axis="0 0 1" range="-2.0 2.0"/>
          <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.028" rgba="0.3 0.6 0.9 1" contype="0" conaffinity="0"/>

          <body name="palm" pos="0.35 0 0">
            <geom name="palm_geom" type="sphere" size="0.028" rgba="0.15 0.15 0.15 1"/>
            <site name="gripper_center" type="sphere" pos="0 0 0" size="0.01" rgba="1 1 1 0.001"/>

            <body name="finger_left" pos="0.05 0 0">
              <joint name="finger_left" type="slide" axis="0 1 0" range="0.02 0.08"/>
              <geom type="capsule" fromto="0 0 0 0 0.055 0" size="0.012" rgba="0.1 0.1 0.1 1"/>
            </body>

            <body name="finger_right" pos="0.05 0 0">
              <joint name="finger_right" type="slide" axis="0 -1 0" range="0.02 0.08"/>
              <geom type="capsule" fromto="0 0 0 0 -0.055 0" size="0.012" rgba="0.1 0.1 0.1 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="puck" pos="0.62 0 0">
      <joint name="puck_x" type="slide" axis="1 0 0" range="0.38 0.90" damping="0.4"/>
      <joint name="puck_y" type="slide" axis="0 1 0" range="-0.28 0.28" damping="0.4"/>
      <geom name="puck_geom" type="cylinder" size="0.03 0.03" mass="0.08" rgba="0.85 0.35 0.25 1"/>
    </body>
  </worldbody>

  <actuator>
    <position joint="shoulder" ctrlrange="-1.4 1.4" kp="60"/>
    <position joint="elbow" ctrlrange="-2.0 2.0" kp="60"/>
    <position joint="finger_left" ctrlrange="0.02 0.08" kp="40"/>
    <position joint="finger_right" ctrlrange="0.02 0.08" kp="40"/>
  </actuator>
</mujoco>
"""


@dataclass
class PushEnvConfig:
    control_dt: float = 0.05
    physics_dt: float = 0.005
    episode_steps: int = 60
    success_distance: float = 0.06
    finger_open: float = 0.055
    goal_x_offset_range: Tuple[float, float] = (0.14, 0.22)
    goal_y_offset_range: Tuple[float, float] = (-0.08, 0.08)


class SimpleTwoFingerPushEnv:
    def __init__(self, config: Optional[PushEnvConfig] = None, seed: int = 0) -> None:
        self.config = config or PushEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(MUJOCO_PUSH_XML)
        self.model.opt.timestep = self.config.physics_dt
        self.data = mujoco.MjData(self.model)
        self.frame_skip = max(1, int(round(self.config.control_dt / self.config.physics_dt)))
        self.elapsed_steps = 0
        self.goal = np.array([0.78, 0.0], dtype=np.float32)

        self._actuated_joint_names = ["shoulder", "elbow", "finger_left", "finger_right"]
        self._qpos_indices = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self._actuated_joint_names],
            dtype=np.int32,
        )
        self._qvel_indices = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self._actuated_joint_names],
            dtype=np.int32,
        )
        self._puck_x_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        ]
        self._puck_y_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        ]
        self._puck_x_qvel = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        ]
        self._puck_y_qvel = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        ]
        self._goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self._gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        self.action_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.action_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.link_lengths = np.array([0.45, 0.35], dtype=np.float32)
        self.home_qpos = np.array([-0.6, 1.25, self.config.finger_open, self.config.finger_open], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.elapsed_steps = 0

        self.data.qpos[self._qpos_indices] = self.home_qpos
        self.data.qvel[:] = 0.0

        puck_xy, goal_xy = self._sample_task()
        self.data.qpos[self._puck_x_qpos] = puck_xy[0]
        self.data.qpos[self._puck_y_qpos] = puck_xy[1]
        self.goal = goal_xy.astype(np.float32)
        self.model.site_pos[self._goal_site_id] = np.array([goal_xy[0], goal_xy[1], 0.0], dtype=np.float64)
        self.data.ctrl[:] = self.home_qpos

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_distance = self.box_goal_distance()
        prev_gripper_xy = self.get_gripper_xy()
        prev_puck_xy = self.get_puck_xy()
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), self.action_low, self.action_high)
        self.data.ctrl[:] = clipped_action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._apply_simplified_push(prev_gripper_xy=prev_gripper_xy, prev_puck_xy=prev_puck_xy)
        self.elapsed_steps += 1
        obs = self._get_obs()
        info = self._build_info()
        current_distance = info["box_goal_distance"]
        reward = float(prev_distance - current_distance)
        if info["success"]:
            reward += 5.0

        terminated = bool(info["success"])
        truncated = self.elapsed_steps >= self.config.episode_steps
        return obs, reward, terminated, truncated, info

    def expert_action(self) -> np.ndarray:
        puck_xy = self.get_puck_xy()
        goal_xy = self.goal
        gripper_xy = self.get_gripper_xy()

        direction = goal_xy - puck_xy
        goal_distance = float(np.linalg.norm(direction))
        if goal_distance < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / goal_distance

        desired_gripper = puck_xy - direction * 0.09
        lateral_error = abs(self._cross_2d(direction, gripper_xy - puck_xy))
        along_error = float(np.dot(gripper_xy - puck_xy, direction))

        if lateral_error > 0.03 or along_error > -0.035:
            target_xy = desired_gripper
        else:
            target_xy = puck_xy + direction * 0.015

        target_xy = np.array(
            [
                np.clip(target_xy[0], 0.18, 0.78),
                np.clip(target_xy[1], -0.28, 0.28),
            ],
            dtype=np.float32,
        )
        q1, q2 = self.inverse_kinematics(target_xy)
        return np.array([q1, q2, self.config.finger_open, self.config.finger_open], dtype=np.float32)

    def inverse_kinematics(self, target_xy: np.ndarray) -> Tuple[float, float]:
        target = np.asarray(target_xy, dtype=np.float32)
        radius = float(np.linalg.norm(target))
        min_radius = abs(float(self.link_lengths[0] - self.link_lengths[1])) + 0.02
        max_radius = float(self.link_lengths.sum()) - 0.02
        if radius < 1e-6:
            target = np.array([min_radius, 0.0], dtype=np.float32)
            radius = min_radius
        clamped_radius = np.clip(radius, min_radius, max_radius)
        target = target * (clamped_radius / radius)

        l1, l2 = self.link_lengths
        cos_q2 = float(
            np.clip((target[0] ** 2 + target[1] ** 2 - l1**2 - l2**2) / (2.0 * l1 * l2), -1.0, 1.0)
        )
        q2 = float(np.arccos(cos_q2))
        k1 = float(l1 + l2 * cos_q2)
        k2 = float(l2 * np.sin(q2))
        q1 = float(np.arctan2(target[1], target[0]) - np.arctan2(k2, k1))
        q1 = float(np.clip(q1, self.action_low[0], self.action_high[0]))
        q2 = float(np.clip(q2, self.action_low[1], self.action_high[1]))
        return q1, q2

    def get_gripper_xy(self) -> np.ndarray:
        return self.data.site_xpos[self._gripper_site_id, :2].astype(np.float32).copy()

    def get_puck_xy(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self._puck_x_qpos],
                self.data.qpos[self._puck_y_qpos],
            ],
            dtype=np.float32,
        )

    def box_goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.get_puck_xy()))

    def _sample_task(self) -> Tuple[np.ndarray, np.ndarray]:
        puck_xy = np.array(
            [
                self.rng.uniform(0.52, 0.64),
                self.rng.uniform(-0.10, 0.10),
            ],
            dtype=np.float32,
        )
        goal_xy = puck_xy + np.array(
            [
                self.rng.uniform(*self.config.goal_x_offset_range),
                self.rng.uniform(*self.config.goal_y_offset_range),
            ],
            dtype=np.float32,
        )
        goal_xy[0] = np.clip(goal_xy[0], 0.68, 0.86)
        goal_xy[1] = np.clip(goal_xy[1], -0.18, 0.18)
        return puck_xy, goal_xy

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._qpos_indices].astype(np.float32)
        qvel = self.data.qvel[self._qvel_indices].astype(np.float32)
        gripper_xy = self.get_gripper_xy()
        puck_xy = self.get_puck_xy()
        obs = np.concatenate(
            [
                qpos,
                qvel,
                gripper_xy,
                puck_xy,
                self.goal.astype(np.float32),
                puck_xy - gripper_xy,
                self.goal.astype(np.float32) - puck_xy,
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _build_info(self) -> Dict:
        box_goal_distance = self.box_goal_distance()
        return {
            "goal_xy": self.goal.astype(np.float32).copy(),
            "puck_xy": self.get_puck_xy(),
            "gripper_xy": self.get_gripper_xy(),
            "box_goal_distance": box_goal_distance,
            "success": bool(box_goal_distance <= self.config.success_distance),
        }

    def _apply_simplified_push(self, prev_gripper_xy: np.ndarray, prev_puck_xy: np.ndarray) -> None:
        gripper_xy = self.get_gripper_xy()
        gripper_delta = gripper_xy - prev_gripper_xy
        goal_direction = self.goal - prev_puck_xy
        goal_distance = float(np.linalg.norm(goal_direction))
        if goal_distance < 1e-6:
            return

        goal_unit = goal_direction / goal_distance
        contact_distance = float(np.linalg.norm(gripper_xy - prev_puck_xy))
        lateral_error = abs(self._cross_2d(goal_unit, gripper_xy - prev_puck_xy))
        behind_score = float(np.dot(prev_puck_xy - gripper_xy, goal_unit))
        forward_motion = float(np.dot(gripper_delta, goal_unit))
        if contact_distance > 0.09 or lateral_error > 0.07 or behind_score < -0.01 or forward_motion <= 0.0:
            return

        tangential_motion = gripper_delta - forward_motion * goal_unit
        puck_delta = goal_unit * min(0.028, 0.95 * forward_motion) + 0.15 * tangential_motion
        new_puck_xy = self.get_puck_xy() + puck_delta.astype(np.float32)
        new_puck_xy[0] = np.clip(new_puck_xy[0], 0.38, 0.90)
        new_puck_xy[1] = np.clip(new_puck_xy[1], -0.28, 0.28)
        self.data.qpos[self._puck_x_qpos] = float(new_puck_xy[0])
        self.data.qpos[self._puck_y_qpos] = float(new_puck_xy[1])
        self.data.qvel[self._puck_x_qvel] = 0.0
        self.data.qvel[self._puck_y_qvel] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])


def build_push_env_from_config(config: Dict, seed: Optional[int] = None) -> SimpleTwoFingerPushEnv:
    env_cfg = config.get("environment", {})
    push_cfg = PushEnvConfig(
        control_dt=float(env_cfg.get("control_dt", 0.05)),
        physics_dt=float(env_cfg.get("physics_dt", 0.005)),
        episode_steps=int(env_cfg.get("episode_steps", 60)),
        success_distance=float(env_cfg.get("success_distance", 0.06)),
        finger_open=float(env_cfg.get("finger_open", 0.055)),
        goal_x_offset_range=tuple(env_cfg.get("goal_x_offset_range", [0.14, 0.22])),
        goal_y_offset_range=tuple(env_cfg.get("goal_y_offset_range", [-0.08, 0.08])),
    )
    seed_value = config.get("experiment", {}).get("seed", 0) if seed is None else seed
    return SimpleTwoFingerPushEnv(config=push_cfg, seed=int(seed_value))

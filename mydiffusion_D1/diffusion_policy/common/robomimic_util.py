from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from scipy.spatial.transform import Rotation


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:
        return None


class AbsoluteActionConversionError(RuntimeError):
    def __init__(
        self,
        *,
        task_name: str,
        dataset_path: str,
        stage: str,
        message: str,
        env_name: str | None = None,
        demo_idx: int | None = None,
        step_idx: int | None = None,
        robot_idx: int | None = None,
        controller_summary: Dict[str, Any] | None = None,
        next_action: str | None = None,
        original_exc: Exception | None = None,
    ) -> None:
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.stage = stage
        self.message = message
        self.env_name = env_name
        self.demo_idx = demo_idx
        self.step_idx = step_idx
        self.robot_idx = robot_idx
        self.controller_summary = controller_summary or {}
        self.next_action = next_action
        self.original_exc = original_exc
        super().__init__(self.format_message())

    def format_message(self) -> str:
        lines = [
            "MimicGen absolute-action 转换失败。",
            f"task={self.task_name}",
            f"dataset={self.dataset_path}",
            f"stage={self.stage}",
        ]
        if self.env_name is not None:
            lines.append(f"env_name={self.env_name}")
        if self.demo_idx is not None:
            lines.append(f"demo_idx={self.demo_idx}")
        if self.step_idx is not None:
            lines.append(f"step_idx={self.step_idx}")
        if self.robot_idx is not None:
            lines.append(f"robot_idx={self.robot_idx}")
        if self.controller_summary:
            lines.append(f"controller={self.controller_summary}")
        lines.append(f"reason={self.message}")
        if self.original_exc is not None:
            lines.append(
                f"cause={type(self.original_exc).__name__}: {self.original_exc}"
            )
        if self.next_action is not None:
            lines.append(f"next={self.next_action}")
        return "\n".join(lines)


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path: str, algo_name: str = "bc", task_name: str | None = None):
        self.dataset_path = str(dataset_path)
        self.task_name = task_name or Path(dataset_path).stem
        self.file: h5py.File | None = None
        self.env = None
        self.abs_env = None

        config = config_factory(algo_name=algo_name)
        ObsUtils.initialize_obs_utils_with_config(config)

        self.env_meta = self._load_env_meta()
        self.env_name = str(self.env_meta.get("env_name", "<unknown>"))
        self.abs_env_meta = self._build_absolute_env_meta()

        self.env = self._create_env(
            self.env_meta,
            stage="create_delta_env",
            next_action="确认 MimicGen 任务已成功注册，并检查 raw HDF5 里的 env metadata 是否可被当前 robosuite / robomimic 版本加载。",
        )
        self.abs_env = self._create_env(
            self.abs_env_meta,
            stage="create_absolute_env",
            next_action="优先检查 env metadata 里的 controller 配置；如果当前任务依赖特殊 controller，请确认 control_delta=False 后仍能被当前 robosuite 版本实例化。",
        )
        self.controller_summary = self._build_controller_summary(self.abs_env_meta, self.abs_env)

        runtime_use_delta = self.controller_summary.get("runtime_use_delta", [])
        if any(value is True for value in runtime_use_delta):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="validate_absolute_env",
                message="absolute-action 环境已创建，但至少有一个 runtime controller 仍然处于 delta 模式。",
                env_name=self.env_name,
                controller_summary=self.controller_summary,
                next_action="检查 env metadata 里的 controller_configs 是否覆盖了 control_delta=False。",
            )

        try:
            self.file = h5py.File(self.dataset_path, "r")
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="open_dataset",
                message="无法打开 raw MimicGen HDF5 数据集。",
                env_name=self.env_name,
                controller_summary=self.controller_summary,
                next_action="确认输入文件存在且不是半写入文件，再重新运行 inspect / convert 流程。",
                original_exc=exc,
            ) from exc

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        for env_name in ("env", "abs_env"):
            env = getattr(self, env_name, None)
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
                setattr(self, env_name, None)

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._data_group())

    def describe_environment(self) -> Dict[str, Any]:
        env_kwargs = self.env_meta.get("env_kwargs", {})
        return {
            "task_name": self.task_name,
            "dataset_path": self.dataset_path,
            "env_name": self.env_name,
            "env_kwargs_keys": sorted(env_kwargs.keys()) if isinstance(env_kwargs, dict) else [],
            "controller_summary": self.controller_summary,
            "demos": len(self),
        }

    def diagnose_demo(self, idx: int, *, run_eval: bool = False) -> Dict[str, Any]:
        states, actions, demo = self._load_demo_arrays(idx)
        abs_actions = self.convert_actions(states, actions, demo_idx=idx)
        summary = {
            "task_name": self.task_name,
            "env_name": self.env_name,
            "dataset_path": self.dataset_path,
            "demo_idx": idx,
            "num_steps": int(actions.shape[0]),
            "action_shape": list(actions.shape),
            "state_shape": list(states.shape),
            "controller_summary": self.controller_summary,
            "abs_action_shape": list(abs_actions.shape),
        }
        if run_eval:
            _, info = self.convert_and_eval_idx(idx)
            summary["eval"] = info
        if "obs" in demo:
            summary["obs_keys"] = sorted(list(demo["obs"].keys()))
        return summary

    def convert_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        demo_idx: int | None = None,
    ) -> np.ndarray:
        expected_action_dim = self._expected_action_dim()
        if actions.ndim != 2:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="validate_actions",
                message=f"actions 必须是二维 [T, D]，当前 shape={actions.shape}。",
                env_name=self.env_name,
                demo_idx=demo_idx,
                controller_summary=self.controller_summary,
                next_action="确认 raw HDF5 的 demo/actions 仍然保持单臂或双臂低维动作格式。",
            )
        if actions.shape[-1] != expected_action_dim:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="validate_actions",
                message=f"动作维度与 runtime robot 数量不匹配，预期 {expected_action_dim}，实际 {actions.shape[-1]}。",
                env_name=self.env_name,
                demo_idx=demo_idx,
                controller_summary=self.controller_summary,
                next_action="先运行 inspect_mimicgen_dataset.py，确认该任务的 raw actions 维度与单臂/双臂配置一致。",
            )
        if len(states) != len(actions):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="validate_demo_lengths",
                message=f"states 与 actions 时间长度不一致，states={len(states)}，actions={len(actions)}。",
                env_name=self.env_name,
                demo_idx=demo_idx,
                controller_summary=self.controller_summary,
                next_action="检查 raw demo 是否完整；当前 replay 逻辑要求每个 step 都能执行 reset_to(state[i]) 后再重放 action[i]。",
            )

        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)
        env = self.env

        action_goal_pos = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[..., [-1]]

        for step_idx in range(len(states)):
            self._reset_env_to_state(env, states[step_idx], demo_idx=demo_idx, step_idx=step_idx)

            for robot_idx, robot in enumerate(env.env.robots):
                action_vec = stacked_actions[step_idx, robot_idx]
                try:
                    robot.control(action_vec, policy_step=True)
                except Exception as exc:
                    raise AbsoluteActionConversionError(
                        task_name=self.task_name,
                        dataset_path=self.dataset_path,
                        stage="robot_control",
                        message=f"执行 robot.control 失败，action.shape={action_vec.shape}。",
                        env_name=self.env_name,
                        demo_idx=demo_idx,
                        step_idx=step_idx,
                        robot_idx=robot_idx,
                        controller_summary=self.controller_summary,
                        next_action="优先确认该任务在当前 state 下能否被 delta-action 环境正常 reset_to；再检查该 controller 是否接受 7 维单步动作。",
                        original_exc=exc,
                    ) from exc

                controller = robot.controller
                goal_pos = np.asarray(getattr(controller, "goal_pos", None), dtype=np.float32)
                if goal_pos.shape != (3,):
                    raise AbsoluteActionConversionError(
                        task_name=self.task_name,
                        dataset_path=self.dataset_path,
                        stage="read_goal_pos",
                        message=f"controller.goal_pos 形状非法，当前 shape={goal_pos.shape}。",
                        env_name=self.env_name,
                        demo_idx=demo_idx,
                        step_idx=step_idx,
                        robot_idx=robot_idx,
                        controller_summary=self.controller_summary,
                        next_action="检查该 controller 在当前任务里是否会在 policy_step 后生成 3 维末端位置目标。",
                    )

                goal_ori = np.asarray(getattr(controller, "goal_ori", None), dtype=np.float32)
                if goal_ori.size != 9:
                    raise AbsoluteActionConversionError(
                        task_name=self.task_name,
                        dataset_path=self.dataset_path,
                        stage="read_goal_ori",
                        message=f"controller.goal_ori 不是可转成 3x3 rotation matrix 的数组，当前 shape={goal_ori.shape}。",
                        env_name=self.env_name,
                        demo_idx=demo_idx,
                        step_idx=step_idx,
                        robot_idx=robot_idx,
                        controller_summary=self.controller_summary,
                        next_action="检查该 controller 在当前任务里是否正确 materialize 了 orientation goal。",
                    )
                goal_ori = goal_ori.reshape(3, 3)
                try:
                    action_goal_ori[step_idx, robot_idx] = Rotation.from_matrix(goal_ori).as_rotvec()
                except Exception as exc:
                    raise AbsoluteActionConversionError(
                        task_name=self.task_name,
                        dataset_path=self.dataset_path,
                        stage="convert_goal_ori",
                        message="controller.goal_ori 无法转换为 rotation vector。",
                        env_name=self.env_name,
                        demo_idx=demo_idx,
                        step_idx=step_idx,
                        robot_idx=robot_idx,
                        controller_summary=self.controller_summary,
                        next_action="检查 absolute-action controller 生成的姿态矩阵是否数值稳定，必要时先只诊断 demo_0 / 前 10 个 demos。",
                        original_exc=exc,
                    ) from exc

                action_goal_pos[step_idx, robot_idx] = goal_pos

        stacked_abs_actions = np.concatenate(
            [action_goal_pos, action_goal_ori, action_gripper],
            axis=-1,
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx: int) -> np.ndarray:
        states, actions, _ = self._load_demo_arrays(idx)
        return self.convert_actions(states, actions, demo_idx=idx)

    def convert_and_eval_idx(self, idx: int):
        env = self.env
        abs_env = self.abs_env
        states, actions, demo = self._load_demo_arrays(idx)
        eval_skip_steps = 1

        abs_actions = self.convert_actions(states, actions, demo_idx=idx)

        try:
            robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
            robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="load_eval_observations",
                message="无法从 demo/obs 里读取 robot0_eef_pos 或 robot0_eef_quat。",
                env_name=self.env_name,
                demo_idx=idx,
                controller_summary=self.controller_summary,
                next_action="检查 raw demo 的 low-dim obs 是否完整，尤其是 robot0_eef_pos / robot0_eef_quat。",
                original_exc=exc,
            ) from exc

        delta_error_info = self.evaluate_rollout_error(
            env,
            states,
            actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
            task_name=self.task_name,
            dataset_path=self.dataset_path,
            env_name=self.env_name,
            demo_idx=idx,
            controller_summary=self.controller_summary,
            stage_prefix="eval_delta",
        )
        abs_error_info = self.evaluate_rollout_error(
            abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
            task_name=self.task_name,
            dataset_path=self.dataset_path,
            env_name=self.env_name,
            demo_idx=idx,
            controller_summary=self.controller_summary,
            stage_prefix="eval_absolute",
        )

        info = {
            "delta_max_error": delta_error_info,
            "abs_max_error": abs_error_info,
        }
        return abs_actions, info

    def _data_group(self):
        assert self.file is not None
        return self.file["data"]

    def _load_env_meta(self) -> Dict[str, Any]:
        try:
            env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="read_env_metadata",
                message="无法从 raw HDF5 读取 env metadata。",
                next_action="先运行 inspect_mimicgen_dataset.py，确认该任务的 raw HDF5 结构和 env_args 正常。",
                original_exc=exc,
            ) from exc
        if not isinstance(env_meta, dict):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="read_env_metadata",
                message=f"读取到的 env metadata 不是 dict，而是 {type(env_meta).__name__}。",
                next_action="检查当前 robomimic 版本是否与 MimicGen raw 数据兼容。",
            )
        return env_meta

    def _build_absolute_env_meta(self) -> Dict[str, Any]:
        abs_env_meta = copy.deepcopy(self.env_meta)
        env_kwargs = abs_env_meta.setdefault("env_kwargs", {})
        if not isinstance(env_kwargs, dict):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="prepare_absolute_env_metadata",
                message=f"env_kwargs 不是 dict，而是 {type(env_kwargs).__name__}。",
                env_name=self.env_name,
                next_action="检查 raw HDF5 里的 env metadata 序列化是否完整。",
            )
        controller_configs = copy.deepcopy(env_kwargs.get("controller_configs") or {})
        if not isinstance(controller_configs, dict):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="prepare_absolute_env_metadata",
                message=f"controller_configs 不是 dict，而是 {type(controller_configs).__name__}。",
                env_name=self.env_name,
                next_action="检查 raw HDF5 metadata 里的 controller_configs 是否符合当前 robosuite 版本预期。",
            )
        controller_configs["control_delta"] = False
        env_kwargs["controller_configs"] = controller_configs
        abs_env_meta["env_kwargs"] = env_kwargs
        return abs_env_meta

    def _create_env(self, env_meta: Dict[str, Any], *, stage: str, next_action: str):
        try:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=False,
                use_image_obs=False,
            )
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage=stage,
                message="根据 env metadata 构造环境失败。",
                env_name=str(env_meta.get("env_name", "<unknown>")),
                controller_summary=self._metadata_controller_summary(env_meta),
                next_action=next_action,
                original_exc=exc,
            ) from exc

        robot_count = len(getattr(env.env, "robots", []))
        if robot_count not in (1, 2):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage=f"{stage}_validate_robot_count",
                message=f"只支持单臂或双臂任务，当前 runtime robots={robot_count}。",
                env_name=str(env_meta.get("env_name", "<unknown>")),
                controller_summary=self._metadata_controller_summary(env_meta),
                next_action="检查该任务是否仍属于 low-dim 单臂 / 双臂 MimicGen 配置。",
            )
        return env

    def _expected_action_dim(self) -> int:
        robot_count = len(getattr(self.env.env, "robots", []))
        return robot_count * 7

    def _controller_summary_string(self) -> Dict[str, Any]:
        return dict(self.controller_summary)

    def _metadata_controller_summary(self, env_meta: Dict[str, Any]) -> Dict[str, Any]:
        env_kwargs = env_meta.get("env_kwargs", {})
        controller_configs = env_kwargs.get("controller_configs") if isinstance(env_kwargs, dict) else None
        keys = sorted(controller_configs.keys()) if isinstance(controller_configs, dict) else []
        control_delta = controller_configs.get("control_delta") if isinstance(controller_configs, dict) else None
        return {
            "metadata_controller_keys": keys,
            "metadata_control_delta": control_delta,
        }

    def _build_controller_summary(self, env_meta: Dict[str, Any], env) -> Dict[str, Any]:
        metadata = self._metadata_controller_summary(env_meta)
        runtime_types = []
        runtime_use_delta = []
        for robot in getattr(env.env, "robots", []):
            controller = getattr(robot, "controller", None)
            runtime_types.append(type(controller).__name__ if controller is not None else "<missing>")
            runtime_use_delta.append(_safe_bool(getattr(controller, "use_delta", None)))
        metadata["runtime_controller_types"] = runtime_types
        metadata["runtime_use_delta"] = runtime_use_delta
        metadata["runtime_robot_count"] = len(runtime_types)
        return metadata

    def _load_demo_arrays(self, idx: int):
        data_group = self._data_group()
        if idx < 0 or idx >= len(data_group):
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="load_demo",
                message=f"demo_idx 超出范围，当前 idx={idx}，总 demos={len(data_group)}。",
                env_name=self.env_name,
                controller_summary=self._controller_summary_string(),
                next_action="先用 --diagnose 跑最小 demo 索引，确认当前数据集里实际有哪些 demo_* 轨迹。",
            )
        demo_key = f"demo_{idx}"
        if demo_key not in data_group:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="load_demo",
                message=f"未找到 `{demo_key}`。",
                env_name=self.env_name,
                demo_idx=idx,
                controller_summary=self._controller_summary_string(),
                next_action="检查 HDF5 里的 demo 命名是否连续，或先运行 inspect_mimicgen_dataset.py。",
            )
        demo = data_group[demo_key]
        try:
            states = demo["states"][:]
            actions = demo["actions"][:]
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="load_demo",
                message="无法读取 demo 的 states 或 actions。",
                env_name=self.env_name,
                demo_idx=idx,
                controller_summary=self._controller_summary_string(),
                next_action="检查 raw HDF5 是否完整，必要时重新下载该任务数据集。",
                original_exc=exc,
            ) from exc
        return states, actions, demo

    def _reset_env_to_state(self, env, state: np.ndarray, *, demo_idx: int | None, step_idx: int | None) -> None:
        try:
            env.reset_to({"states": state})
        except Exception as exc:
            raise AbsoluteActionConversionError(
                task_name=self.task_name,
                dataset_path=self.dataset_path,
                stage="reset_to_state",
                message=f"env.reset_to 失败，state.shape={np.asarray(state).shape}。",
                env_name=self.env_name,
                demo_idx=demo_idx,
                step_idx=step_idx,
                controller_summary=self._controller_summary_string(),
                next_action="优先用 --diagnose --demo-idx 0 确认最早失败 step，再检查该任务的 raw states 是否与当前环境版本兼容。",
                original_exc=exc,
            ) from exc

    @staticmethod
    def evaluate_rollout_error(
        env,
        states,
        actions,
        robot0_eef_pos,
        robot0_eef_quat,
        metric_skip_steps=1,
        *,
        task_name: str,
        dataset_path: str,
        env_name: str,
        demo_idx: int,
        controller_summary: Dict[str, Any],
        stage_prefix: str,
    ):
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()

        for step_idx in range(len(states)):
            try:
                env.reset_to({"states": states[step_idx]})
            except Exception as exc:
                raise AbsoluteActionConversionError(
                    task_name=task_name,
                    dataset_path=dataset_path,
                    stage=f"{stage_prefix}_reset_to_state",
                    message="评估 replay 时 env.reset_to 失败。",
                    env_name=env_name,
                    demo_idx=demo_idx,
                    step_idx=step_idx,
                    controller_summary=controller_summary,
                    next_action="如果诊断模式已经定位到同一 step，请优先检查该任务 raw states 与当前 robosuite 版本的兼容性。",
                    original_exc=exc,
                ) from exc
            try:
                env.step(actions[step_idx])
                obs = env.get_observation()
                rollout_next_states.append(env.get_state()["states"])
                rollout_next_eef_pos.append(obs["robot0_eef_pos"])
                rollout_next_eef_quat.append(obs["robot0_eef_quat"])
            except Exception as exc:
                raise AbsoluteActionConversionError(
                    task_name=task_name,
                    dataset_path=dataset_path,
                    stage=f"{stage_prefix}_step",
                    message=f"评估 replay 时 env.step 或 get_observation 失败，action.shape={np.asarray(actions[step_idx]).shape}。",
                    env_name=env_name,
                    demo_idx=demo_idx,
                    step_idx=step_idx,
                    controller_summary=controller_summary,
                    next_action="先跑 --diagnose 缩小到同一个 demo / step，再确认该任务在 delta 与 absolute 环境下都能完成单步重放。",
                    original_exc=exc,
                ) from exc

        rollout_next_states = np.asarray(rollout_next_states)
        rollout_next_eef_pos = np.asarray(rollout_next_eef_pos)
        rollout_next_eef_quat = np.asarray(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) * Rotation.from_quat(
            rollout_next_eef_quat[:-1]
        ).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        return {
            "state": float(max_next_state_diff),
            "pos": float(max_next_eef_pos_dist),
            "rot": float(max_next_eef_rot_dist),
        }

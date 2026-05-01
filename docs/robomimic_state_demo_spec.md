# robomimic Low-Dim Demonstration NPZ 规范

这份说明对应当前推荐的纯仿真正式 baseline 路线：

- 主数据来源：`robomimic` HDF5 demonstration
- 主输入：`low-dim state`
- 内部训练格式：当前项目统一使用 `.npz`

## 推荐工作流

1. 准备一个 robomimic demonstration HDF5 文件。
2. 选定要拼接的 low-dim observation 键，保持固定顺序。
3. 用转换脚本把 HDF5 转成当前项目可直接训练的 `.npz`。
4. 用 `inspect_demo.py` 检查输出形状，再开始训练。

## 转换命令

```bash
python scripts/convert_robomimic_hdf5_to_npz.py \
  --input-hdf5 /path/to/lift/ph/low_dim_v141.hdf5 \
  --obs-keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object \
  --output-npz data/robomimic_lift_state_smoke/lift_low_dim.npz
```

## 输出 NPZ 最小字段

转换后的 `.npz` 至少包含：

- `observations`
- `actions`
- `episode_lengths`

其中：

- `observations.shape = [num_episodes, max_episode_len, obs_dim]`
- `actions.shape = [num_episodes, max_episode_len, action_dim]`
- `episode_lengths.shape = [num_episodes]`

这里的 `max_episode_len` 是 padding 后的统一长度，真实长度由 `episode_lengths` 指定。
当前项目的读取器会按 `episode_lengths` 截断后再切训练窗口，不会把 padding 区域当成有效数据。

## 推荐 low-dim obs 键

以 `Lift` 任务为起点时，推荐从这组键开始：

- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- `object`

它们会按给定顺序展平并拼接成单个状态向量。
如果你换了任务，`obs_dim` 必须以真实数据转换结果为准。

## 配置对应关系

推荐先使用：

- [configs/robomimic_lift_state_smoke.yaml](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/configs/robomimic_lift_state_smoke.yaml)

关键字段：

- `data.demo_root`
- `data.source_format`
- `data.source_obs_keys`
- `data.flatten_obs`
- `data.obs_dim`
- `data.action_dim`

## 检查与训练

检查转换结果：

```bash
python scripts/inspect_demo.py --config configs/robomimic_lift_state_smoke.yaml
```

训练：

```bash
python scripts/train.py --config configs/robomimic_lift_state_smoke.yaml
```

离线评估：

```bash
python scripts/evaluate.py \
  --config configs/robomimic_lift_state_smoke.yaml \
  --checkpoint results/robomimic_lift_state_smoke/model_final.pt
```

导出最小版 rollout 视频：

```bash
python scripts/export_robomimic_rollout_video.py \
  --config configs/robomimic_lift_state_smoke.yaml \
  --checkpoint results/robomimic_lift_state_smoke/model_final.pt \
  --output-video results/robomimic_lift_state_smoke/robomimic_rollout.mp4
```

## 当前边界

当前这条 robomimic 路线已经支持：

- `HDF5 -> npz` 转换
- `inspect_demo`
- `offline train`
- `offline eval`
- 最小版单条 rollout 视频导出

当前这条视频导出路径仍然是最小版：

- 默认面向 `Lift + low-dim`
- 主要目标是导出单条 `mp4`
- 还没有做标准 benchmark 级别的多 rollout 评测汇总

# PickCube-v1 + state Demonstration NPZ 规范

这份规范是当前推荐给你的第一版正式 baseline 数据规范。

目标：

- 任务固定为 `PickCube-v1`
- 观测模式固定为 `state`
- demonstration 保存为 `.npz`
- 让 `full_baseline_project` 可以直接读取

## 为什么先选这套规范

因为它最适合新手起步：

- 任务简单
- 数据结构清楚
- 不需要先处理图像
- 更容易确认模型输入输出是否正确

## 推荐字段

每个 `.npz` 至少包含：

### `observations`

推荐 shape：

```text
[num_episodes, episode_len, obs_dim]
```

含义：

- `num_episodes`：多少条 episode
- `episode_len`：每条轨迹长度
- `obs_dim`：状态观测维度

如果你暂时每个文件只存一条轨迹，也可以先用：

```text
[episode_len, obs_dim]
```

项目里的读取器会自动升成 `[1, episode_len, obs_dim]`。

### `actions`

推荐 shape：

```text
[num_episodes, episode_len, action_dim]
```

或单条轨迹：

```text
[episode_len, action_dim]
```

## 推荐附加字段

这些字段当前不是训练必需，但非常建议保存：

- `env_states`
- `success`
- `rewards`
- `dones`
- `episode_lengths`
- `task_name`
- `obs_mode`
- `control_mode`

## 最小可用版本

如果你只是想先让 baseline 吃到真实 demonstration，最小集合就是：

- `observations`
- `actions`

## 当前 baseline 如何使用这些数据

项目不会直接用 `actions` 当输入，而是会在训练前自动构造：

- `target_action = actions`
- `noisy_action = actions + noise`
- `diffusion_step = sampled integer`

所以 `.npz` 里不需要你提前存 `noisy_action`。

## 示例 shape

例如：

- `num_episodes = 20`
- `episode_len = 50`
- `obs_dim = 32`
- `action_dim = 8`

那么：

- `observations.shape = [20, 50, 32]`
- `actions.shape = [20, 50, 8]`

## 和当前项目配置的对应关系

在 `configs/pickcube_state_demo_template.yaml` 里，对应的关键字段是：

- `demo_root`
- `obs_key`
- `action_key`
- `sequence_length`
- `stride`
- `obs_dim`
- `action_dim`

## 当前推荐配置

- 任务：`PickCube-v1`
- 观测模式：`state`
- 控制模式：`pd_ee_delta_pose`
- sequence_length：`16`
- stride：`4`

## 重要提醒

这里的 `obs_dim=32` 和 `action_dim=8` 只是起步模板。

真实维度必须以你最终实际采集出来的数据为准。
因此推荐你在拿到第一批 demonstrations 后，先运行：

```bash
python scripts/inspect_demo.py --config configs/pickcube_state_demo_template.yaml
```

然后再回头修正配置里的 `obs_dim` 和 `action_dim`。

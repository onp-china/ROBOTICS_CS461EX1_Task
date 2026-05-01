# Diffusion Policy Baseline Project

这是一个面向课程实验的 baseline 工程，核心目标是用统一的训练主干快速验证 `Diffusion Transformer Policy` 在不同仿真数据来源上的表现。

当前推荐路线已经调整为：

- 正式实验优先走 `robomimic low-dim state -> npz -> offline train / eval`
- 当前自定义 `MuJoCo two-finger push` 保留为 smoke / 回归测试
- `synthetic` 数据继续保留为最小可运行教学入口

项目只做 baseline，不包含 `AttnRes`。

## 当前能力

- 训练教学版 `Diffusion Transformer Policy`
- 保存配置、指标、曲线、checkpoint、summary
- 支持 `synthetic` 数据 smoke
- 支持读取统一格式的 demonstration `.npz`
- 支持把 `robomimic` low-dim HDF5 转成当前项目使用的 `.npz`
- 支持把 replay 后的 ManiSkill `trajectory.h5` 转成当前项目使用的 `.npz`
- 支持 MuJoCo push smoke 的数据生成、离线评估和闭环 rollout
- 支持 `episode_lengths`，会在切窗前自动去掉 padding 区域

## 当前推荐的三条路线

### 1. `synthetic`

用途：

- 最快确认训练脚本、模型、结果落盘是否正常

特点：

- 不依赖真实 demonstration
- 最适合第一次跑通工程

### 2. `robomimic low-dim state`

用途：

- 当前推荐的正式仿真 baseline 主线

特点：

- 更接近标准 manipulation benchmark
- 直接服务于后续 diffusion policy 实验
- 当前已经支持 `HDF5 -> npz -> inspect -> offline train -> offline eval`
- 当前还没有接 robomimic 标准环境的闭环 rollout bridge

### 3. `MuJoCo two-finger push smoke`

用途：

- 本地回归测试
- 演示从数据生成到闭环 rollout 的完整链路

特点：

- 环境简单、可控
- 不是标准 benchmark
- 更适合教学和链路验证，不建议作为正式主实验环境

## 目录结构

```text
ROBOTICS_CS461EX1_Task/
├── configs/
│   ├── baseline_maniskill_template.yaml
│   ├── baseline_synthetic.yaml
│   ├── mujoco_two_finger_push_smoke.yaml
│   ├── pickcube_state_demo_template.yaml
│   └── robomimic_lift_state_smoke.yaml
├── docs/
│   ├── migration_notes.md
│   ├── pickcube_state_demo_spec.md
│   └── robomimic_state_demo_spec.md
├── notebooks/
├── scripts/
│   ├── convert_maniskill_h5_to_npz.py
│   ├── convert_robomimic_hdf5_to_npz.py
│   ├── evaluate.py
│   ├── evaluate_push_rollout.py
│   ├── export_robomimic_rollout_video.py
│   ├── generate_mujoco_push_demos.py
│   ├── inspect_demo.py
│   ├── show_config.py
│   └── train.py
├── src/
│   ├── adapters/
│   │   ├── demo_npz.py
│   │   ├── maniskill_stub.py
│   │   └── robomimic_hdf5.py
│   ├── data/
│   ├── envs/
│   ├── eval/
│   ├── models/
│   ├── trainers/
│   └── utils/
├── README.md
└── README_MUJOCO.md
```

## 安装

推荐使用仓库内的 `robot` 虚拟环境，或者自己创建一个新 venv。

```bash
cd /Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task
python -m venv robot
source robot/bin/activate
pip install -r requirements.txt
```

如果你主要跑 MuJoCo smoke，也可以继续使用：

```bash
pip install -r requirements_mujoco.txt
```

如果你要导出 `robomimic low-dim` 的 rollout 视频，还需要额外依赖：

```bash
pip install -r requirements_robomimic_rollout.txt
```

## 快速开始

### A. 最小 smoke：`synthetic`

查看配置：

```bash
python scripts/show_config.py --config configs/baseline_synthetic.yaml
```

训练：

```bash
python scripts/train.py --config configs/baseline_synthetic.yaml
```

评估：

```bash
python scripts/evaluate.py \
  --config configs/baseline_synthetic.yaml \
  --checkpoint results/full_baseline_synthetic/model_final.pt
```

### B. 推荐主线：`robomimic low-dim`

1. 把 robomimic HDF5 转成项目统一的 `.npz`

```bash
python scripts/convert_robomimic_hdf5_to_npz.py \
  --input-hdf5 /path/to/low_dim_v141.hdf5 \
  --obs-keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object \
  --output-npz data/robomimic_lift_state_smoke/lift_low_dim.npz
```

2. 检查转换结果

```bash
python scripts/inspect_demo.py --config configs/robomimic_lift_state_smoke.yaml
```

3. 训练

```bash
python scripts/train.py --config configs/robomimic_lift_state_smoke.yaml
```

4. 离线评估

```bash
python scripts/evaluate.py \
  --config configs/robomimic_lift_state_smoke.yaml \
  --checkpoint results/robomimic_lift_state_smoke/model_final.pt
```

5. 导出最小版 rollout 视频

```bash
python scripts/export_robomimic_rollout_video.py \
  --config configs/robomimic_lift_state_smoke.yaml \
  --checkpoint results/robomimic_lift_state_smoke/model_final.pt \
  --output-video results/robomimic_lift_state_smoke/robomimic_rollout.mp4
```

更详细的数据规范见：

- [robomimic_state_demo_spec.md](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/docs/robomimic_state_demo_spec.md)

### C. MuJoCo push smoke

1. 生成 demonstration

```bash
python scripts/generate_mujoco_push_demos.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

2. 检查数据

```bash
python scripts/inspect_demo.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

3. 训练

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python scripts/train.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

4. 离线评估

```bash
python scripts/evaluate.py \
  --config configs/mujoco_two_finger_push_smoke.yaml \
  --checkpoint results/mujoco_two_finger_push_smoke/model_final.pt
```

5. 闭环 rollout

```bash
python scripts/evaluate_push_rollout.py \
  --config configs/mujoco_two_finger_push_smoke.yaml \
  --checkpoint results/mujoco_two_finger_push_smoke/model_final.pt
```

MuJoCo 细节说明见：

- [README_MUJOCO.md](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/README_MUJOCO.md)

## 数据格式

当前训练主干统一读取 `.npz`，最小字段为：

- `observations`
- `actions`

推荐附加字段：

- `episode_lengths`
- `success`
- `rewards`
- `task_name`
- `obs_mode`
- `control_mode`

支持两种基本形状：

- `observations/actions: [T, D]`
- `observations/actions: [E, T, D]`

如果是 padding 后堆叠的多条轨迹，建议同时提供：

- `episode_lengths: [E]`

当前读取器会先按 `episode_lengths` 截断每条轨迹，再切训练窗口。

## 关键配置

### `configs/baseline_synthetic.yaml`

用途：

- 工程最小可运行入口

### `configs/robomimic_lift_state_smoke.yaml`

用途：

- 当前推荐的纯仿真 diffusion policy 起点配置

关键点：

- 任务：`Lift`
- 输入：`low_dim state`
- 数据来源：`robomimic HDF5 -> npz`
- 当前范围：`offline train + offline eval`

### `configs/mujoco_two_finger_push_smoke.yaml`

用途：

- 回归测试和闭环链路验证

关键点：

- 任务：自定义 `two_finger_push`
- 数据来源：项目内部生成的 `.npz`
- 支持闭环 rollout

### `configs/pickcube_state_demo_template.yaml`

用途：

- 如果后续要继续尝试 ManiSkill / PickCube state demonstration，可以以这套模板为起点

### `configs/baseline_maniskill_template.yaml`

用途：

- 保留为 demonstration `.npz` 模板配置

## 当前状态与边界

- 当前最稳的正式实验入口是 `robomimic low-dim -> npz -> offline train / eval`
- 当前已经支持 robomimic low-dim 的最小版单条 rollout 视频导出
- 当前还没有做标准 benchmark 级别的多 rollout 评测与汇总
- MuJoCo push smoke 已经能跑通完整闭环链路，但不是标准 benchmark
- 当前仓库依然是为课程实验整理的本地 baseline 工程，不是官方仓库原样复现

MuJoCo smoke 的代表性结果：

- demonstration expert success rate: `0.825`
- offline validation mse: `0.00446`
- offline validation success rate: `0.9918`
- rollout success rate: `0.20`

这也说明它更适合作为回归测试，而不是最终 benchmark。

## 关键文件

- 数据读取：[src/adapters/demo_npz.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/src/adapters/demo_npz.py)
- robomimic 转换：[src/adapters/robomimic_hdf5.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/src/adapters/robomimic_hdf5.py)
- MuJoCo 环境：[src/envs/mujoco_push_env.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/src/envs/mujoco_push_env.py)
- 训练入口：[scripts/train.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/train.py)
- 离线评估入口：[scripts/evaluate.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/evaluate.py)

## 相关文档

- [migration_notes.md](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/docs/migration_notes.md)
- [pickcube_state_demo_spec.md](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/docs/pickcube_state_demo_spec.md)
- [robomimic_state_demo_spec.md](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/docs/robomimic_state_demo_spec.md)

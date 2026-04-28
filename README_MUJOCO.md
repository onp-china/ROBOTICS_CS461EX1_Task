# MuJoCo Two-Finger Push Baseline

这个文档说明当前仓库中的 MuJoCo 版本 baseline。

当前方案的目标是：

- 使用一个简化的 MuJoCo 两指夹爪机械臂 `push` 任务
- 用 demonstration 数据训练教学版 diffusion policy
- 完成数据生成、训练、离线评估、闭环 rollout 测试

## 目录

- 环境定义：[src/envs/mujoco_push_env.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/src/envs/mujoco_push_env.py)
- 数据生成脚本：[scripts/generate_mujoco_push_demos.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/generate_mujoco_push_demos.py)
- 训练脚本：[scripts/train.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/train.py)
- 离线评估脚本：[scripts/evaluate.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/evaluate.py)
- 闭环 rollout 评估脚本：[scripts/evaluate_push_rollout.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/scripts/evaluate_push_rollout.py)
- 当前 smoke 配置：[configs/mujoco_two_finger_push_smoke.yaml](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/configs/mujoco_two_finger_push_smoke.yaml)

## 依赖安装

推荐使用你当前已经创建好的虚拟环境：

```bash
source robot/bin/activate
pip install -r requirements_mujoco.txt
```

## 运行流程

### 1. 生成 demonstration 数据

```bash
source robot/bin/activate
python scripts/generate_mujoco_push_demos.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

生成结果默认会写到：

```text
data/mujoco_two_finger_push_smoke_demos
```

### 2. 检查数据

```bash
source robot/bin/activate
python scripts/inspect_demo.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

### 3. 训练 diffusion policy

```bash
source robot/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib
python scripts/train.py \
  --config configs/mujoco_two_finger_push_smoke.yaml
```

### 4. 离线评估

```bash
source robot/bin/activate
python scripts/evaluate.py \
  --config configs/mujoco_two_finger_push_smoke.yaml \
  --checkpoint results/mujoco_two_finger_push_smoke/model_final.pt
```

### 5. 闭环 rollout 测试

```bash
source robot/bin/activate
python scripts/evaluate_push_rollout.py \
  --config configs/mujoco_two_finger_push_smoke.yaml \
  --checkpoint results/mujoco_two_finger_push_smoke/model_final.pt
```

当前配置默认会在 rollout 时固定夹爪保持张开，更适合 `push` 任务。

## 输出结果

训练输出目录：

```text
results/mujoco_two_finger_push_smoke
```

主要文件包括：

- `metrics.csv`
- `training_curve.png`
- `summary.json`
- `model_final.pt`

## 当前结果

基于当前 smoke 配置，一次完整实验的代表性结果如下：

- demonstration expert success rate: `0.825`
- offline validation mse: `0.00446`
- offline validation success rate: `0.9918`
- rollout success rate: `0.20`

## 说明

- 这是一个可运行的 MuJoCo smoke baseline，重点是验证 diffusion policy 训练和测试链路。
- 当前环境是简化任务，不是标准 benchmark 复现。
- 离线拟合已经比较稳定，但闭环 rollout 成功率还不高，后续仍有提升空间。

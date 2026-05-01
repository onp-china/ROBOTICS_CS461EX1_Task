# MimicGen D1 `low_dim_abs` Self-Contained Baseline

这个目录是独立的 D1 训练工程：

- 只支持 `core D1`
- 只支持 `low_dim_abs`
- 训练入口固定为 `python mydiffusion_D1/train.py ...`
- 训练所需的最小 `diffusion_policy` 子集已经内置到 `mydiffusion_D1/diffusion_policy/`
- 默认训练配置先禁用 rollout，优先保证离线训练链可直接跑通

当前支持的任务：

- `mug_cleanup_d1`
- `coffee_d1`
- `three_piece_assembly_d1`

## 设计说明

这个工程保留了 D0 版本的 [mydiffusion](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/mydiffusion) 作为历史版本，不会修改它。

`mydiffusion_D1` 是一套新的、自包含的 D1 baseline。它不再依赖仓库根目录的外置 `diffusion_policy/` 源码目录，运行时优先使用：

- `mydiffusion_D1/diffusion_policy/`

其中包含从上游 `diffusion_policy` 裁剪出来并在本地维护的最小训练子集。上游许可证为 MIT，许可证文本保存在：

- [LICENSE](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/mydiffusion_D1/LICENSE)

## 推荐环境

- macOS Apple Silicon
- `python3.11`
- 本地虚拟环境：`/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/robot311`
- `mujoco==2.3.2`
- `robosuite==1.4.1`
- `robomimic==0.2.0`
- 本地 MimicGen 仓库：`/Users/zzzgys/Desktop/robot/mimicgen`

## 安装

```bash
cd /Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task
python3.11 -m venv robot311
source robot311/bin/activate
python -m pip install -U pip setuptools wheel
brew install cmake ffmpeg
python -m pip install -r mydiffusion_D1/requirements_mimicgen.txt
python -m pip install --no-deps robomimic==0.2.0
python -m pip install robosuite==1.4.1
python -m pip install -e /Users/zzzgys/Desktop/robot/mimicgen
```

如果你已经装过不兼容的 `robosuite`，先卸载再装：

```bash
python -m pip uninstall -y robosuite
python -m pip install robosuite==1.4.1
```

## 数据准备

1. 下载 D1 原始数据

```bash
python mydiffusion_D1/scripts/download_mimicgen_datasets.py \
  --mimicgen-repo /Users/zzzgys/Desktop/robot/mimicgen
```

2. 检查数据并刷新 D1 task yaml

```bash
python mydiffusion_D1/scripts/inspect_mimicgen_dataset.py
```

这一步会：

- 生成 `mydiffusion_D1/reports/dataset_manifest.json`
- 刷新 `mydiffusion_D1/diffusion_policy/config/task/*.yaml`
- 把真实 `obs_dim`、`obs_keys`、`max_steps` 写入 D1 task config

3. 转 absolute action

```bash
python mydiffusion_D1/scripts/convert_mimicgen_abs_actions.py
```

如果 `coffee_d1` 或 `three_piece_assembly_d1` 在 absolute-action 转换时失败，推荐固定按下面顺序排查：

```bash
python mydiffusion_D1/scripts/inspect_mimicgen_dataset.py
python mydiffusion_D1/scripts/convert_mimicgen_abs_actions.py --task coffee_d1 --diagnose --demo-idx 0
python mydiffusion_D1/scripts/convert_mimicgen_abs_actions.py --task three_piece_assembly_d1 --diagnose --demo-idx 0
python mydiffusion_D1/scripts/convert_mimicgen_abs_actions.py --task coffee_d1 --diagnose --max-demos 10
python mydiffusion_D1/scripts/convert_mimicgen_abs_actions.py --task three_piece_assembly_d1 --diagnose --max-demos 10
```

诊断输出会把失败区分为三类：

- 任务未注册或环境构造失败
- absolute-controller 环境构造失败
- demo replay、`robot.control`、`controller.goal_*` 读取失败

`--diagnose` 不会写任何 processed HDF5。
如果不带 `--diagnose` 且传了 `--demo-idx` 或 `--max-demos`，脚本会生成局部调试产物，不会覆盖正式的 `processed/<task>/low_dim_abs.hdf5`。

## 训练

单任务训练：

```bash
python mydiffusion_D1/train.py \
  --config-name=train_diffusion_transformer_mimicgen_d1_lowdim_abs_workspace \
  task=mug_cleanup_d1_lowdim_abs \
  training.seed=42 \
  training.device=mps
```

默认训练不会在训练过程中自动创建 rollout 环境，但当前配置会在训练结束后自动优先根据 rollout 表现挑选 checkpoint：

- 更高的 `test/contact_rate`
- 更小的 `test/mean_min_eef_object_distance`
- 更大的 `test/mean_object_displacement`
- 再回退参考 `test/mean_score` 与 `val_loss`

然后额外导出一条 rollout 视频到：

- `mydiffusion_D1/outputs/<task_config>/<seed>/best_epoch_rollout.mp4`
- `mydiffusion_D1/outputs/<task_config>/<seed>/curves/<task_config>_seed<seed>_loss_curve.png`
- `mydiffusion_D1/outputs/<task_config>/<seed>/curves/<task_config>_seed<seed>_action_mse_curve.png`

如果你想关闭这一步，可在训练命令里覆盖：

```bash
training.auto_export_best_rollout_video=False
training.auto_export_training_curves=False
```

手动闭环 rollout 与视频导出仍然可以单独运行：

```bash
python mydiffusion_D1/scripts/export_rollout_video.py \
  --task mug_cleanup_d1_lowdim_abs \
  --checkpoint /path/to/checkpoints/latest.ckpt
```

如果当前机器没有 MPS，可改成：

```bash
python mydiffusion_D1/train.py \
  --config-name=train_diffusion_transformer_mimicgen_d1_lowdim_abs_workspace \
  task=mug_cleanup_d1_lowdim_abs \
  training.seed=42 \
  training.device=cpu
```

包装脚本：

```bash
python mydiffusion_D1/scripts/run_baseline.py \
  --task mug_cleanup_d1_lowdim_abs \
  --seed 42 \
  --device mps
```

批量运行：

```bash
MYDIFFUSION_D1_DEVICE=mps bash mydiffusion_D1/scripts/run_all_baselines.sh
```

## 结果汇总

```bash
python mydiffusion_D1/scripts/collect_baseline_results.py
```

输出：

- `mydiffusion_D1/reports/baseline_summary.csv`
- `mydiffusion_D1/reports/baseline_summary.md`

## 训练结束后的产物

单次训练 run 默认会在：

- `mydiffusion_D1/outputs/<task_config>/<seed>/`

下保留这些结果：

- `logs.json.txt`
  - 训练过程中的逐步日志
  - 常见字段包括 `train_loss`、`val_loss`、`train_action_mse_error`
- `checkpoints/`
  - `latest.ckpt`
  - 按 `val_loss` 保留的 top-k checkpoint
- `wandb/`
  - offline wandb 记录

汇总脚本会额外生成：

- `mydiffusion_D1/reports/baseline_summary.csv`
- `mydiffusion_D1/reports/baseline_summary.md`

## 生成图表

先汇总结果：

```bash
python mydiffusion_D1/scripts/collect_baseline_results.py
```

再生成图表：

```bash
python mydiffusion_D1/scripts/plot_results.py
```

默认会把图输出到：

- `mydiffusion_D1/reports/figures/curves/`
- `mydiffusion_D1/reports/figures/aggregates/`
- `mydiffusion_D1/reports/figures/contact_sheets/`

生成内容包括：

- 单次训练曲线
  - `train_loss / val_loss`
  - `train_action_mse_error`
  - 如果日志里包含 rollout 分数，也会额外画出 `test/mean_score`
- 多 seed 聚合图
  - 同一任务的最佳 checkpoint 指标
  - 同一任务的 `final val_loss` 均值与标准差
- 跨任务柱状图
  - 三个 D1 任务的最佳 checkpoint 指标
  - 三个 D1 任务的 `final val_loss`
- rollout 视频抽帧拼图
  - 会同时检查训练期 `media/` 和单独导出脚本生成的 `rollout_export/media/`

### 抽帧拼图是什么

抽帧拼图是把一条 rollout 视频中的关键时刻截成静态图，再横向拼成一张图，用于报告或论文展示。

当前默认规则固定为 4 帧：

- 起点
- `1/3`
- `2/3`
- 终点

也就是你会得到类似下面这种静态展示：

- `reset -> 中途阶段 1 -> 中途阶段 2 -> 最终结果`

它比直接插入 mp4 更适合写实验报告，也更方便比较不同任务、不同 seed、不同模型版本。

## 目录约定

```text
mydiffusion_D1/
├── diffusion_policy/
├── scripts/
├── data/
├── outputs/
├── reports/
└── wandb/
```

其中：

- `data/mimicgen/raw/core/<task>.hdf5` 是原始 D1 数据
- `data/mimicgen/processed/<task>/low_dim_abs.hdf5` 是 absolute action 转换后的数据
- `reports/dataset_manifest.json` 记录数据检查结果
- `outputs/<task_config>/<seed>/` 是训练输出

## 后续 residual attention 改动入口

后续如果要在 Transformer 中加入 residual attention，主改动入口固定为：

- [transformer_for_diffusion.py](/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/mydiffusion_D1/diffusion_policy/model/diffusion/transformer_for_diffusion.py)

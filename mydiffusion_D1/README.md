# MimicGen D1 `low_dim_abs` Self-Contained Baseline

这个目录是独立的 D1 训练工程：

- 只支持 `core D1`
- 只支持 `low_dim_abs`
- 训练入口固定为 `python mydiffusion_D1/train.py ...`
- 训练所需的最小 `diffusion_policy` 子集已经内置到 `mydiffusion_D1/diffusion_policy/`

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

## 训练

单任务训练：

```bash
python mydiffusion_D1/train.py \
  --config-name=train_diffusion_transformer_mimicgen_d1_lowdim_abs_workspace \
  task=mug_cleanup_d1_lowdim_abs \
  training.seed=42 \
  training.device=mps
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

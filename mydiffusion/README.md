# MimicGen D0 `low_dim_abs` Baseline

这个目录只放课程实验里新增的 MimicGen baseline 文件，不修改上游
`/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/diffusion_policy` 代码。

## 目标

- 基线模型固定为上游 `DiffusionTransformerLowdimPolicy`
- 任务固定为：
  - `mug_cleanup_d0`
  - `coffee_d0`
  - `three_piece_assembly_d0`
- 观测模式固定为 `low_dim_abs`

## 当前建议环境

下面这套是按你当前机器状态整理过的可运行组合：

- macOS Apple Silicon
- `python3.11`
- 本地虚拟环境：`/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task/robot311`
- `mujoco==2.3.2`
- `robosuite==1.4.1`
- `robomimic==0.2.0`
- 本地 MimicGen 仓库：`/Users/zzzgys/Desktop/robot/mimicgen`

重要兼容性说明：

- MimicGen 官方文档明确说明当前代码不支持 `robosuite v1.5+`。
- 在 macOS 上直接 `pip install robomimic==0.2.0` 往往会卡在 `egl_probe`。当前建议做法是先装本文件列出的依赖，再用 `--no-deps` 安装 `robomimic`。
- `mujoco==2.3.2` 安装后如果出现 `mink requires mujoco>=3.1.6`，这是版本冲突提示，不会回滚 MuJoCo。若想去掉这条提示，可以执行 `python -m pip uninstall -y mink`。

## 一次性安装

```bash
cd /Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task
python3.11 -m venv robot311
source robot311/bin/activate
python -m pip install -U pip setuptools wheel
brew install cmake ffmpeg
python -m pip install -r mydiffusion/requirements_mimicgen.txt
python -m pip install --no-deps robomimic==0.2.0
python -m pip install robosuite==1.4.1
python -m pip install -e /Users/zzzgys/Desktop/robot/mimicgen
```

如果你已经装过不兼容的 `robosuite`，先卸载再装：

```bash
python -m pip uninstall -y robosuite
python -m pip install robosuite==1.4.1
```

如果想完全对齐 MimicGen 文档，也可以把 `robosuite` 换成官方测试 commit 的源码安装；`v1.4.1` 是当前更省事的选择。

## 安装自检

```bash
python - <<'PY'
import torch
import robomimic
import robosuite
import mimicgen
import mujoco
import robosuite.environments.manipulation.single_arm_env
print("torch ok", torch.__version__)
print("robomimic ok")
print("robosuite ok", getattr(robosuite, "__version__", "unknown"))
print("mimicgen ok")
print("mujoco ok", mujoco.__version__)
print("single_arm_env ok")
PY
```

如果最后能看到 `single_arm_env ok`，说明 MimicGen 需要的 robosuite 任务模块已经就位。

## 下载完成后能不能直接训练

不能。

下载脚本只会拿到原始 HDF5。训练前还必须完成这两步：

1. `inspect_mimicgen_dataset.py`
2. `convert_mimicgen_abs_actions.py`

只有下面三类文件都存在时，才算真正具备训练条件：

- `mydiffusion/data/mimicgen/raw/core/<task>.hdf5`
- `mydiffusion/reports/dataset_manifest.json`
- `mydiffusion/data/mimicgen/processed/<task>/low_dim_abs.hdf5`

## 推荐执行顺序

1. 下载原始数据

```bash
python mydiffusion/scripts/download_mimicgen_datasets.py --mimicgen-repo /Users/zzzgys/Desktop/robot/mimicgen
```

2. 检查数据并固化 task yaml

```bash
python mydiffusion/scripts/inspect_mimicgen_dataset.py
```

这一步会：

- 读取 3 个 HDF5 的环境元信息和 observation 键
- 生成 `mydiffusion/reports/dataset_manifest.json`
- 自动刷新 `mydiffusion/config/task/*.yaml`

3. 转 absolute action

```bash
python mydiffusion/scripts/convert_mimicgen_abs_actions.py
```

4. 单任务 smoke

Apple Silicon 建议显式指定 `mps`：

```bash
python mydiffusion/train.py task=mug_cleanup_d0_lowdim_abs training.seed=42 training.device=mps
```

如果 `mps` 后续遇到 PyTorch 算子兼容问题，再退回：

```bash
python mydiffusion/train.py task=mug_cleanup_d0_lowdim_abs training.seed=42 training.device=cpu
```

5. 单任务包装入口

```bash
python mydiffusion/scripts/run_baseline.py --task mug_cleanup_d0_lowdim_abs --seed 42 --device mps
```

6. 完整 baseline

```bash
MYDIFFUSION_DEVICE=mps bash mydiffusion/scripts/run_all_baselines.sh
```

7. 汇总结果

```bash
python mydiffusion/scripts/collect_baseline_results.py
```

## 训练输出与可视化

- 训练日志会写到 `mydiffusion/outputs/<task>/<seed>/logs.json.txt`
- checkpoint 会写到同一目录
- offline wandb 日志会写到 `mydiffusion/wandb/`
- rollout 评测视频会写到每个 run 目录下的 `media/` 子目录

也就是说，最终结果视频是可以直接保存下来的，不只是在终端里看到一个分数。

## 目录

```text
mydiffusion/
├── config/
├── data/
├── outputs/
├── reports/
└── scripts/
```

其中：

- `data/mimicgen/raw/core/<task>.hdf5` 是原始下载数据
- `data/mimicgen/processed/<task>/low_dim_abs.hdf5` 是 absolute action 转换后的数据
- `reports/dataset_manifest.json` 记录数据检查结果
- `outputs/<task_config>/<seed>/` 是训练输出

## 约定

- 训练前必须先跑一次 `inspect_mimicgen_dataset.py`，因为它会把真实 `obs_dim`、`obs_keys`、`max_steps` 写回 task yaml。
- `low_dim_abs.hdf5` 在磁盘上保持单臂 7 维绝对动作；模型侧 `action_dim=10` 是上游数据集类把旋转从 axis-angle 转成 6D representation 后的训练维度。
- 如果环境元信息里找不到可靠的 horizon / max steps，系统会回退到 `1000`，并在 `dataset_manifest.json` 中记录该回退。

# Full Baseline Project

这是一个面向正式实验过渡的 baseline 项目。

它的设计目标有两层：

1. 当前可以在本地用 `synthetic` 数据完整跑通 baseline 训练、评估和结果输出。
2. 后续可以较平滑地接入真实 `ManiSkill` / demonstration 数据，继续作为正式 baseline 项目使用。

这个项目已经在 Diffusion Transformer Policy 的 block 残差路径中集成了 `AttnRes`，并保留了与 baseline 一致的训练与评估入口。

## 当前能力

- 训练一个教学版 baseline `Diffusion Transformer Policy`
- 保存配置、指标、曲线、checkpoint、summary
- 支持 `synthetic` 数据模式
- 预留 `maniskill_demo_npz` 和 `maniskill_stub` 接口
- 支持从 `.npz` demonstration 轨迹切出训练窗口

## 目录结构

```text
full_baseline_project/
├── configs/
│   ├── baseline_synthetic.yaml
│   ├── baseline_maniskill_template.yaml
│   └── pickcube_state_demo_template.yaml
├── docs/
│   ├── migration_notes.md
│   └── pickcube_state_demo_spec.md
├── notebooks/
├── results/
├── requirements.txt
├── scripts/
│   ├── evaluate.py
│   ├── show_config.py
│   └── train.py
└── src/
    ├── adapters/
    │   ├── demo_npz.py
    │   └── maniskill_stub.py
    ├── data/
    │   ├── builder.py
    │   └── synthetic.py
    ├── eval/
    │   ├── evaluator.py
    │   ├── metrics.py
    │   └── visualize.py
    ├── models/
    │   └── diffusion_transformer_policy.py
    ├── trainers/
    │   └── baseline_trainer.py
    └── utils/
        ├── config.py
        ├── io.py
        └── seed.py
```

## 安装

```bash
cd /Users/kiki/Documents/Codex/2026-04-22-files-mentioned-by-the-user-proposal/full_baseline_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

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

检查 demonstration 数据：

```bash
python scripts/inspect_demo.py --config configs/baseline_maniskill_template.yaml
```

把 replay 后的 ManiSkill `trajectory.h5` 转成 baseline `.npz`：

```bash
python scripts/convert_maniskill_h5_to_npz.py \
  --input-h5 /path/to/trajectory.state.pd_ee_delta_pose.h5 \
  --output-npz /path/to/pickcube_state_demo.npz
```

## 两套配置的区别

### `baseline_synthetic.yaml`

这是当前可直接运行的版本，适合学习 baseline 和验证训练逻辑。

### `baseline_maniskill_template.yaml`

这是未来正式版本的模板配置。
只要你把 `demo_root` 改成真实 `.npz` demonstrations 所在目录，并保证字段名匹配，就可以直接走项目的数据读取逻辑。

当前优先支持两种输入形状：

- `[num_episodes, episode_len, dim]`
- `[num_steps, dim]`

### `pickcube_state_demo_template.yaml`

这是我推荐你作为第一版真实 baseline 的起点配置：

- 任务：`PickCube-v1`
- 观测：`state`
- demonstration：`.npz`

建议你优先围绕这套配置准备数据。

## 之后接真实 baseline 要改哪里

重点文件：

- `src/adapters/maniskill_stub.py`
- `src/adapters/demo_npz.py`
- `src/data/builder.py`
- `src/models/diffusion_transformer_policy.py`

## 说明

当前项目依然不是官方仓库的原样复现，而是一个为正式实验准备的本地 baseline 工程。

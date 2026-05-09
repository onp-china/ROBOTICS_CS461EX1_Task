# Three Piece Assembly — AttnRes 对照实验

## 项目结构

```
mydiffusion/
├── core/
│   ├── training/__init__.py    # DiffusionTrainer（标准 DDPM 训练）
│   ├── models/__init__.py     # build_model(variant, ...) 构造模型
│   └── data/__init__.py       # load_mimicgen_lowdim / compute_cond_dim
├── scripts/
│   ├── experiments/
│   │   ├── configs.py          # 7 个实验配置（D0~D4 × baseline/attn_res）
│   │   └── run_experiment.py   # 训练 + rollout 一次搞定（CLI 入口）
│   └── evaluation/
│       └── eval_core_ckpt.py   # rollout 评估脚本（支持 history buffer）
├── experiment.ipynb              # 主实验 notebook（改 EXP_ID 切换实验）
├── EXPERIMENTS_PLAN.md          # 4 人分工表
└── data/mimicgen/processed/three_piece_assembly_d0/low_dim_abs.hdf5
```

## 快速开始

### 1. 环境安装

```bash
# 基础依赖
pip install -r requirements.txt

# robomimic / mimicgen（按需）
pip install --no-deps robomimic==0.2.0
pip install robosuite==1.4.1
pip install -e <your_mimicgen_path>
```

### 2. 运行第一个实验

```bash
# 在服务器上
cd /root/mydiffusion

# 查看可用实验
python scripts/experiments/run_experiment.py --list

# 跑 D0_attn_res（150 epoch + rollout）
python scripts/experiments/run_experiment.py --exp-id D0_attn_res

# 跑 D0_baseline
python scripts/experiments/run_experiment.py --exp-id D0_baseline
```

结果自动写入 `outputs_compare/{exp_id}/`。

---

## Notebook 使用方法（推荐）

1. 在服务器上打开 `experiment.ipynb`
2. **只改 cell 9 的 `EXP_ID`**（例如 `'D0_baseline'`、`'D0_attn_res'`）
3. 跑 cell 9（显示配置）
4. 跑 cell 20（训练 + rollout，自动完成）
5. 跑 cell 26（查看 train_loss 曲线和 rollout score）

切换实验只需改 cell 9 的 `EXP_ID`，其他不用动。

---

## 实验配置一览

| ID | variant | n_obs_steps | n_layer | n_head | obs_keys | use_history | 说明 |
|---|---|---|---|---|---|---|---|
| D0_baseline | baseline | 2 | 8 | 4 | 默认 4 项 | ❌ | 标准 baseline，阶段 1 必跑 |
| D0_attn_res | attn_res | 2 | 8 | 4 | 默认 4 项 | ✅ | 历史 buffer，阶段 1 核心对照 |
| D1_baseline | baseline | 16 | 8 | 4 | 默认 4 项 | ❌ | 长 cond baseline |
| D1_attn_res | attn_res | 16 | 8 | 4 | 默认 4 项 | ✅ | 长 cond + 历史 buffer |
| D2_attn_res | attn_res | 16 | 12 | 4 | 默认 4 项 | ✅ | 更深的 attn_res |
| D3_attn_res | attn_res | 16 | 8 | 8 | 默认 4 项 | ✅ | 更多 head 的 attn_res |
| D4_attn_res | attn_res | 16 | 8 | 4 | +joint_vel | ✅ | 更丰富 obs 的 attn_res |

> `use_history=True` 的实验在 rollout 时维护滚动 obs history buffer，模型每步能看到从第 0 步到当前步的全部历史。

---

## 实验流程

### 阶段 1（必跑）

4 人分工，每人跑 1~2 组：

| 人 | 任务 |
|---|---|
| A | D0_baseline → D1_baseline |
| B | D0_attn_res → D1_attn_res |
| C | D2_attn_res → D3_attn_res |
| D | D4_attn_res |

每组 150 epoch，约 2~3.5 小时（含 rollout）。

**决策点**：阶段 1 跑完后汇总 `test/mean_score`，判断 attn_res 是否显著超过 baseline，再决定是否跑阶段 2。

### 阶段 2（看阶段 1 结果决定）

在阶段 1 基础上探索 attn_res 更受益的条件（D2/D3/D4）。

---

## 实验参数说明

| 参数 | 说明 |
|---|---|
| `n_obs_steps` | 每次给模型的 obs 帧数（rollout 时由环境决定） |
| `use_history` | 推理时是否维护滚动 history buffer（训练时不用） |
| `attnres_blend_init` | attn_res 路径的初始 gate（0.5=有效，0.0≈不工作） |
| `attnres_temperature` | attn_res mixer 的 softmax 温度 |

---

## 常见问题

**Q: rollout 分数为 0？**
- 确认 `n_obs_steps` 是否与训练一致（环境只有 2 帧 obs，训练用 16 帧就会 0 分）
- 确认 `scripts/experiments/run_experiment.py` 是最新版本（git pull）
- 先跑 5 epoch 的 dry-run 验证管线：`--num-epochs 5`

**Q: 训练 loss 不下降？**
- 确认 ckpt 里 `normalizer_state_dict` 和 `noise_scheduler_cfg` 存在（新版 trainer 才会写入）

**Q: 怎么加新的实验配置？**
- 编辑 `scripts/experiments/configs.py`，在 `EXPERIMENTS` dict 里加一行
- 然后 `python scripts/experiments/run_experiment.py --list` 确认出现

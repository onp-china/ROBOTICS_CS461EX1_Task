# Migration Notes

这个项目当前处在“正式 baseline 的本地过渡版”阶段。

## 当前已经完整的部分

- 配置体系
- baseline 模型
- 训练器
- 评估器
- 可视化
- 结果落盘逻辑

## 需要你以后替换的部分

### 1. 数据读取

从：

- `synthetic`

替换成：

- `maniskill_demo_npz`
- 或未来你采用的其他 demonstration / replay buffer 格式

重点文件：

- `src/adapters/demo_npz.py`
- `src/adapters/maniskill_stub.py`
- `src/data/builder.py`

### 2. 模型实现

当前是教学版 baseline `Diffusion Transformer Policy`。

如果你要更接近官方仓库，需要把：

- `src/models/diffusion_transformer_policy.py`

替换成官方实现或按官方结构重写。

### 3. 评估指标

当前 success rate 和 cumulative reward 是教学版代理指标。

正式版要替换成：

- 100 次 rollout 的真实 success rate
- episode return
- 训练过程中的定期 eval

## 当前 demonstration 读取器已经支持什么

`src/adapters/demo_npz.py` 当前支持：

- 读取一个目录中的多个 `.npz` 文件
- 自动识别 `[E, T, D]` 或 `[T, D]` 形式的轨迹
- 把长轨迹切成固定 `sequence_length` 的训练窗口
- 自动生成 `noisy_action` 和 `diffusion_step`

## 你接真实数据时最常改的字段

- `demo_root`
- `obs_key`
- `action_key`
- `sequence_length`
- `stride`

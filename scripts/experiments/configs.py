"""
scripts/experiments/configs.py

多组对照实验配置。每组都是 (variant, model_cfg, data_cfg) 三元组。
cond_dim 在 run_experiment.py 里根据 obs_keys 自动算，这里写不写都不影响。

实验设计：
  阶段 1（核心对照，必跑）：
    D0_baseline / D0_attn_res
        标准 diffusion_policy 设置：n_obs_steps=2, n_layer=8, n_head=4
        目的：先证明 attn_res ≥ baseline
  阶段 2（规律探索，看阶段 1 结果决定是否跑）：
    D1：n_obs_steps 2→16     看长 condition 是否让 attn_res 更受益
    D2：D1 + n_layer 8→12    看更深时 attn_res 是否进一步领先
    D3：D1 + n_head 4→8      看更多 head 时 attn_res 是否进一步领先
    D4：D1 + 加 joint_vel    看更丰富 obs 时 attn_res 是否进一步领先
"""
from __future__ import annotations

DEFAULT_OBS_KEYS = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
D4_OBS_KEYS = DEFAULT_OBS_KEYS + ["robot0_joint_vel"]

# 共享训练超参
SHARED_TRAINING = dict(
    num_epochs=150,
    lr=1e-4,
    weight_decay=1e-3,
    betas=(0.9, 0.95),
    amp_dtype="tf32",
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    batch_size=256,
    grad_accum_steps=1,
    use_ema=True,
    ema_decay=0.999,
    grad_clip=1.0,
    log_every=10,
    val_every=1,
    checkpoint_every=50,
    seed=42,
)

# 共享 model 基础（每组按需 override）
_MODEL_BASE = dict(
    input_dim=10,
    output_dim=10,
    horizon=16,
    n_layer=8,
    n_head=4,
    n_emb=256,
    p_drop_emb=0.0,
    p_drop_attn=0.3,
    causal_attn=True,
    obs_as_cond=True,
    time_as_cond=True,
    n_cond_layers=0,
    attnres_last_bias_init=8.0,
    attnres_temperature=2.0,
    attnres_blend_init=0.5,
)

# 共享 data 基础
_DATA_BASE = dict(
    horizon=16,
    n_action_steps=8,
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    val_ratio=0.02,
)


def _make(variant: str, n_obs_steps: int, n_layer: int, n_head: int,
          obs_keys: list[str], use_history: bool = False) -> dict:
    """组装一组实验的完整配置。cond_dim 由 runner 自动算。"""
    return dict(
        variant=variant,
        model_cfg={
            **_MODEL_BASE,
            "n_obs_steps": n_obs_steps,
            "n_layer": n_layer,
            "n_head": n_head,
        },
        data_cfg={
            **_DATA_BASE,
            "n_obs_steps": n_obs_steps,
            "obs_keys": list(obs_keys),
        },
        use_history=use_history,
    )


# ---------------------------------------------------------------------------
# 实验矩阵
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, dict] = {
    # ===== 阶段 1：核心对照 =====
    # D0_attn_res 用 use_history=True，推理时把历史 obs 也拼给模型
    "D0_baseline":  _make("baseline", n_obs_steps=2,  n_layer=8,  n_head=4, obs_keys=DEFAULT_OBS_KEYS, use_history=False),
    "D0_attn_res":  _make("attn_res",  n_obs_steps=2,  n_layer=8,  n_head=4, obs_keys=DEFAULT_OBS_KEYS, use_history=True),

    # ===== 阶段 2：规律探索（看阶段 1 结果再决定是否跑）=====
    "D1_baseline":  _make("baseline", n_obs_steps=16, n_layer=8,  n_head=4, obs_keys=DEFAULT_OBS_KEYS, use_history=False),
    "D1_attn_res":  _make("attn_res",  n_obs_steps=16, n_layer=8,  n_head=4, obs_keys=DEFAULT_OBS_KEYS, use_history=True),

    "D2_attn_res":  _make("attn_res",  n_obs_steps=16, n_layer=12, n_head=4, obs_keys=DEFAULT_OBS_KEYS, use_history=True),
    "D3_attn_res":  _make("attn_res",  n_obs_steps=16, n_layer=8,  n_head=8, obs_keys=DEFAULT_OBS_KEYS, use_history=True),
    "D4_attn_res":  _make("attn_res",  n_obs_steps=16, n_layer=8,  n_head=4, obs_keys=D4_OBS_KEYS,  use_history=True),
}


def list_experiments() -> list[str]:
    return list(EXPERIMENTS.keys())


def get_experiment(exp_id: str) -> dict:
    if exp_id not in EXPERIMENTS:
        raise KeyError(
            f"未知实验 ID {exp_id!r}。可用：{list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[exp_id]

#!/usr/bin/env python3
"""
eval_core_ckpt.py

用 DiffusionTrainer (core/training) 存的 checkpoint 跑仿真 rollout 评估。
绕开 Hydra/train.py，直接实例化 RobomimicLowdimRunner + 自定义 policy。

使用方法（model_cfg 已写入 ckpt，无需手动传超参）：
    python scripts/evaluation/eval_core_ckpt.py \
        --ckpt outputs_notebook/checkpoints/epoch=0199.ckpt \
        --out-dir outputs_notebook \
        --n-test 4 --n-test-vis 1

如果需要临时改某个超参：
    --override-cfg n_layer=12 attnres_temperature=4.0
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_POLICY_ROOT = Path("/root/diffusion_policy")

for p in (REPO_ROOT, DIFFUSION_POLICY_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class CoreModelPolicy:
    """
    把 DiffusionTrainer 的模型包成 RobomimicLowdimRunner 能调用的 policy。

    Option A（历史 buffer）：
        reset() 初始化一个空的 obs_history（list）
        每步 predict_action() 收到新 obs，追加到 history
        cond = 全部 history（从第 0 步到当前步的完整 obs），超过 horizon 截断
        action 切片不变，仍然取前 n_action_steps 步执行

    训练时 cond = nobs[:, :n_obs_steps]（固定 2 步），
    推理时 cond = 全部 history（越来越长）——
    这里训练和推理的 cond 维度不同，
    但模型通过 attention 学习到跨时间步的依赖，推理时能利用更丰富的历史。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,
        noise_scheduler,
        n_obs_steps: int,
        n_action_steps: int,
        horizon: int,
        action_dim: int,
        device: str = "cuda:0",
        num_inference_steps: int = 100,
        use_history: bool = True,
    ):
        self.model = model
        self.normalizer = normalizer
        self.noise_scheduler = noise_scheduler
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.use_history = use_history
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.model.eval()
        self.model.to(self.device)
        if self.normalizer is not None:
            self.normalizer.to(self.device)
        self.obs_history: list = []

    def reset(self):
        """每个 episode 开始时清空 history buffer。"""
        self.obs_history.clear()

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> dict:
        obs = obs_dict["obs"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device).float()
        B = obs.shape[0]

        # 追加新 obs 到 history
        self.obs_history.append(obs[0])  # [cond_dim]

        if self.use_history:
            # Option A: cond = 全部 history（截取最近 horizon 步，不够则 pad）
            hist = self.obs_history[-self.horizon:]
            pad_len = self.horizon - len(hist)
            if pad_len > 0:
                pad = torch.zeros(pad_len, obs.shape[-1], device=self.device)
                hist = [pad] + hist
            cond = torch.stack(hist, dim=0).unsqueeze(0)  # [1, T, cond_dim]
        else:
            # 标准做法：只用最近 n_obs_steps 步
            nobs = self.normalizer["obs"].normalize(obs) if self.normalizer else obs
            cond = nobs[:, : self.n_obs_steps]

        # 初始噪声
        traj = torch.randn(
            B, self.horizon, self.action_dim, device=self.device, dtype=torch.float32
        )

        # 反向去噪
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
            eps = self.model(sample=traj, timestep=t_batch, cond=cond)
            if isinstance(eps, tuple):
                eps = eps[0]
            traj = self.noise_scheduler.step(
                model_output=eps, timestep=int(t), sample=traj
            ).prev_sample

        # 反归一化
        if self.normalizer is not None:
            naction_pred = self.normalizer["action"].unnormalize(traj)
        else:
            naction_pred = traj

        # 切片：取前 n_action_steps 步
        action = naction_pred[:, : self.n_action_steps]

        return {"action": action, "action_pred": naction_pred}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eval DiffusionTrainer checkpoint with RobomimicLowdimRunner"
    )
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--out-dir", required=True, help="Output dir for logs/media")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-test", type=int, default=4)
    parser.add_argument("--n-test-vis", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument(
        "--use-model-weights",
        action="store_true",
        help="强制使用 model_state_dict 而非 EMA 权重",
    )
    parser.add_argument(
        "--override-cfg",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="临时覆盖 ckpt 里的 model_cfg（如: n_layer=12 attnres_temperature=4.0）",
    )
    parser.add_argument(
        "--use-history",
        action="store_true",
        default=False,
        help="推理时用历史 buffer（滚动 obs history）；默认关闭（标准固定 n_obs_steps）",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 checkpoint ----
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
          f"train_loss={ckpt.get('train_loss', float('nan')):.4f}")

    # ---- 2. 从 ckpt 自动读 model_cfg + data_cfg ----
    model_cfg = dict(ckpt.get("model_cfg") or {})
    data_cfg = dict(ckpt.get("data_cfg") or {})

    if not model_cfg:
        print("⚠️  ckpt 不含 model_cfg（旧 checkpoint），回退到默认超参。"
              "建议用新版 DiffusionTrainer 重训以获得自描述 ckpt。")
        model_cfg = dict(
            input_dim=10, output_dim=10, horizon=16, n_obs_steps=16, cond_dim=51,
            n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3,
            causal_attn=True, obs_as_cond=True,
            attnres_last_bias_init=8.0, attnres_temperature=2.0,
        )

    for kv in args.override_cfg:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            v = eval(v, {"__builtins__": {}}, {})
        except Exception:
            pass
        model_cfg[k] = v
        print(f"  override model_cfg.{k} = {v!r}")

    print(f"  model_cfg: {model_cfg}")
    print(f"  data_cfg : {data_cfg or '(empty)'}")

    # ---- 2.5 构建 noise scheduler ----
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    ns_cfg = ckpt.get("noise_scheduler_cfg")
    if ns_cfg:
        ns_cfg = dict(ns_cfg)
        ns_cfg.pop("_use_default_values", None)
        ns_cfg.pop("_class_name", None)
        ns_cfg.pop("_diffusers_version", None)
        noise_scheduler = DDPMScheduler(**ns_cfg)
        num_train_timesteps = noise_scheduler.config.num_train_timesteps
        print(f"  noise_scheduler: DDPMScheduler from ckpt "
              f"(T={num_train_timesteps}, schedule={noise_scheduler.config.beta_schedule}, "
              f"pred={noise_scheduler.config.prediction_type})")
    else:
        # legacy ckpt（线性 alpha_t = 1 - t/100，没有真正的 DDPM 加噪），fallback 用 cosine
        # 注意：这种情况下推理结果不一定好，建议用新版 trainer 重训。
        print("⚠️  ckpt 不含 noise_scheduler_cfg（旧 checkpoint）。"
              "fallback 到 DDPMScheduler(squaredcos_cap_v2, T=100)。"
              "强烈建议用新版 DiffusionTrainer 重训。")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        num_train_timesteps = 100

    # ---- 3. 构建模型（根据 ckpt 里的 model_class 自动选 baseline / attnres）----
    model_class = ckpt.get("model_class", "ResAttentionTransformerForDiffusion")
    model_module = ckpt.get(
        "model_module", "resattention_transformer_for_diffusion"
    )
    print(f"  model_class : {model_class}")
    print(f"  model_module: {model_module}")

    try:
        mod = importlib.import_module(model_module)
        ModelCls = getattr(mod, model_class)
    except (ImportError, AttributeError) as exc:
        # baseline ckpt 但 module 路径写法不同时的兜底
        if "TransformerForDiffusion" in model_class:
            from diffusion_policy.model.diffusion.transformer_for_diffusion import (
                TransformerForDiffusion as ModelCls,
            )
        else:
            from resattention_transformer_for_diffusion import (
                ResAttentionTransformerForDiffusion as ModelCls,
            )
        print(f"  ⚠️ 自动 import 失败 ({exc})，fallback 到 {ModelCls.__name__}")

    # baseline TransformerForDiffusion 不接受 attnres_* 参数
    if "AttnRes" not in ModelCls.__name__ and "ResAttention" not in ModelCls.__name__:
        model_cfg = {k: v for k, v in model_cfg.items() if not k.startswith("attnres_")}

    model = ModelCls(**model_cfg)

    # ---- 4. 加载权重（优先 EMA）----
    ema_state = ckpt.get("ema_state_dict")
    model_state = ckpt.get("model_state_dict")

    if not args.use_model_weights and ema_state is not None:
        print("Applying EMA weights...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name])
        loaded = "ema"
    elif model_state is not None:
        print("Loading model_state_dict...")
        model.load_state_dict(model_state)
        loaded = "model"
    else:
        raise ValueError("Checkpoint has neither ema_state_dict nor model_state_dict")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (loaded from: {loaded})")

    # ---- 4.5 加载 normalizer（关键）----
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    normalizer = None
    norm_state = ckpt.get("normalizer_state_dict")
    if norm_state is not None:
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(norm_state)
        print("Loaded LinearNormalizer from ckpt.")
    else:
        print("⚠️  ckpt 不含 normalizer_state_dict（旧 checkpoint）。"
              "推理时不归一化 obs/action。强烈建议用新版 DiffusionTrainer 重训。")

    # ---- 5. 创建 policy wrapper ----
    horizon = model_cfg.get("horizon", 16)
    n_obs_steps = ckpt.get("n_obs_steps") or model_cfg.get("n_obs_steps", 2)
    action_dim = model_cfg.get("input_dim", 10)
    n_action_steps = data_cfg.get("n_action_steps", 8)

    policy = CoreModelPolicy(
        model=model,
        normalizer=normalizer,
        noise_scheduler=noise_scheduler,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        horizon=horizon,
        action_dim=action_dim,
        device=args.device,
        num_inference_steps=min(args.num_inference_steps, num_train_timesteps),
        use_history=args.use_history,
    )

    # ---- 5. 注册 MimicGen 环境 ----
    for mod_name in ("mimicgen", "mimicgen_envs"):
        try:
            importlib.import_module(mod_name)
            print(f"Registered environments via {mod_name}")
            break
        except ImportError:
            continue

    # ---- 6. 启动 RobomimicLowdimRunner ----
    from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner

    dataset_path = str(
        REPO_ROOT / "data/mimicgen/processed/three_piece_assembly_d0/low_dim_abs.hdf5"
    )
    # obs_keys 从 ckpt 读（D4 等含自定义 obs 的 ckpt 必须靠这个）
    obs_keys = ckpt.get("obs_keys")
    if obs_keys is None:
        # legacy ckpt 兼容
        obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        print(f"⚠️  ckpt 不含 obs_keys，fallback 默认 4 项: {obs_keys}")
    else:
        print(f"obs_keys (from ckpt): {obs_keys}")

    runner = RobomimicLowdimRunner(
        output_dir=str(out_dir),
        dataset_path=dataset_path,
        obs_keys=obs_keys,
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        test_start_seed=100000,
        max_steps=args.max_steps,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        n_latency_steps=0,
        fps=10,
        crf=22,
        past_action=False,
        abs_action=True,
        n_envs=args.n_envs,
    )

    print(f"\nRunning rollout: n_test={args.n_test}, n_test_vis={args.n_test_vis}, "
          f"n_envs={args.n_envs}...")
    runner_log = runner.run(policy)

    print("\nResults:")
    for k, v in runner_log.items():
        print(f"  {k}: {v}")

    # ---- 7. 写日志（过滤不可序列化的 wandb.Video 等对象）----
    def _json_safe(v):
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        return str(v)

    log_path = out_dir / "logs_eval.json.txt"
    record = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": str(ckpt_path),
        "loaded_weights": loaded,
        "model_cfg": model_cfg,
        **{k: _json_safe(v) for k, v in runner_log.items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    score = runner_log.get("test/mean_score")
    print(f"\n✅ test/mean_score = {score}")
    print(f"Log saved: {log_path}")
    if args.n_test_vis > 0:
        media_dir = out_dir / "media"
        if media_dir.is_dir():
            mp4s = list(media_dir.glob("*.mp4"))
            print(f"Videos in {media_dir}: {len(mp4s)} files")


if __name__ == "__main__":
    main()

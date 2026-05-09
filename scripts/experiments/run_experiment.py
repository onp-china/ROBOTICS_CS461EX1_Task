#!/usr/bin/env python3
"""
scripts/experiments/run_experiment.py

按实验 ID 跑单组训练 + 训练结束后的 rollout 评估。
每组结果写入独立目录 outputs_compare/{exp_id}/，互不干扰。

用法（4 台机各跑各的）：
    python scripts/experiments/run_experiment.py --exp-id D0_baseline
    python scripts/experiments/run_experiment.py --exp-id D0_attn_res
    python scripts/experiments/run_experiment.py --exp-id D1_baseline
    ...

可选参数：
    --num-epochs   覆盖默认 150
    --device       默认 cuda:0
    --skip-rollout 训完不做 rollout（debug 用）
    --num-inference-steps   rollout 用的去噪步数（默认 100）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_POLICY_ROOT = Path("/root/diffusion_policy")
for p in (REPO_ROOT, DIFFUSION_POLICY_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    parser = argparse.ArgumentParser(
        description="跑单组对照实验：训练 + 训练后 rollout"
    )
    parser.add_argument("--exp-id", default=None,
                        help="实验 ID，如 D0_baseline / D1_attn_res（见 configs.py）。"
                             "传 --list 时可不填。")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="覆盖默认 num_epochs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs_compare"),
                        help="所有实验的根输出目录")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="只训练不做 rollout（debug 用）")
    parser.add_argument("--num-inference-steps", type=int, default=100,
                        help="rollout 时的去噪步数")
    parser.add_argument("--n-test", type=int, default=4)
    parser.add_argument("--n-test-vis", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="列出全部实验 ID 后退出")
    args = parser.parse_args()

    from scripts.experiments.configs import (
        EXPERIMENTS, get_experiment, SHARED_TRAINING, list_experiments,
    )

    if args.list:
        print("可用实验 ID:")
        for k in list_experiments():
            spec = EXPERIMENTS[k]
            mc = spec["model_cfg"]
            dc = spec["data_cfg"]
            print(f"  {k:14s} variant={spec['variant']:8s} "
                  f"n_obs={dc['n_obs_steps']:2d} n_layer={mc['n_layer']:2d} "
                  f"n_head={mc['n_head']} obs_keys={len(dc['obs_keys'])}")
        return

    if args.exp_id is None:
        parser.error("必须指定 --exp-id（或传 --list 查看可用 ID）")

    exp = get_experiment(args.exp_id)
    variant = exp["variant"]
    model_cfg = dict(exp["model_cfg"])
    data_cfg = dict(exp["data_cfg"])

    # ---- 计算 cond_dim（根据 obs_keys 从 hdf5 读 shape）----
    from core.data import compute_cond_dim, load_mimicgen_lowdim
    dataset_path = REPO_ROOT / "data/mimicgen/processed/three_piece_assembly_d0/low_dim_abs.hdf5"
    obs_keys = data_cfg["obs_keys"]
    cond_dim = compute_cond_dim(dataset_path, obs_keys)
    model_cfg["cond_dim"] = cond_dim

    # ---- 输出目录 ----
    exp_out = Path(args.out_root) / args.exp_id
    exp_out.mkdir(parents=True, exist_ok=True)
    (exp_out / "checkpoints").mkdir(exist_ok=True)

    # ---- 训练参数 ----
    num_epochs = args.num_epochs or SHARED_TRAINING["num_epochs"]

    print("=" * 70)
    print(f"Experiment: {args.exp_id}")
    print(f"  variant     : {variant}")
    print(f"  n_obs_steps : {data_cfg['n_obs_steps']}")
    print(f"  n_layer     : {model_cfg['n_layer']}")
    print(f"  n_head      : {model_cfg['n_head']}")
    print(f"  obs_keys    : {obs_keys}")
    print(f"  cond_dim    : {cond_dim}  (computed from obs_keys)")
    print(f"  num_epochs  : {num_epochs}")
    print(f"  device      : {args.device}")
    print(f"  out_dir     : {exp_out}")
    print("=" * 70)

    # ---- 构建模型 ----
    from core.models import build_model, count_parameters
    if variant == "baseline":
        build_kwargs = {k: v for k, v in model_cfg.items() if not k.startswith("attnres_")}
    else:
        build_kwargs = dict(model_cfg)
    model = build_model(variant, **build_kwargs)
    print(f"模型参数量: {count_parameters(model)/1e6:.2f}M")

    # ---- 加载数据 ----
    print(f"加载数据集: {dataset_path}")
    dataset, train_loader, val_loader = load_mimicgen_lowdim(
        dataset_path=dataset_path,
        horizon=data_cfg["horizon"],
        n_obs_steps=data_cfg["n_obs_steps"],
        n_action_steps=data_cfg["n_action_steps"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        persistent_workers=data_cfg["persistent_workers"],
        prefetch_factor=data_cfg["prefetch_factor"],
        val_ratio=data_cfg["val_ratio"],
        obs_keys=obs_keys,
    )
    print(f"训练集 {len(dataset) - max(1, int(len(dataset)*data_cfg['val_ratio']))} 条 / "
          f"验证集 {max(1, int(len(dataset)*data_cfg['val_ratio']))} 条")

    # ---- 训练 ----
    from core.training import DiffusionTrainer, TrainingConfig
    cfg = TrainingConfig(
        device=args.device,
        num_epochs=num_epochs,
        lr=SHARED_TRAINING["lr"],
        weight_decay=SHARED_TRAINING["weight_decay"],
        betas=SHARED_TRAINING["betas"],
        amp_dtype=SHARED_TRAINING["amp_dtype"],
        cudnn_benchmark=SHARED_TRAINING["cudnn_benchmark"],
        cudnn_deterministic=SHARED_TRAINING["cudnn_deterministic"],
        batch_size=data_cfg["batch_size"],
        grad_accum_steps=SHARED_TRAINING["grad_accum_steps"],
        use_ema=SHARED_TRAINING["use_ema"],
        ema_decay=SHARED_TRAINING["ema_decay"],
        grad_clip=SHARED_TRAINING["grad_clip"],
        log_every=SHARED_TRAINING["log_every"],
        val_every=SHARED_TRAINING["val_every"],
        checkpoint_every=SHARED_TRAINING["checkpoint_every"],
        output_dir=exp_out,
    )
    trainer = DiffusionTrainer(
        model, train_loader, val_loader, cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        dataset=dataset,
        n_obs_steps=data_cfg["n_obs_steps"],
        obs_as_cond=model_cfg["obs_as_cond"],
        obs_keys=obs_keys,
        use_history=exp.get("use_history", False),
    )
    history = trainer.fit(num_epochs=num_epochs)
    print(f"训练完成: train_loss={history['train_loss'][-1]:.4f}, "
          f"val_loss={history['val_loss'][-1]:.4f}")

    # ---- 找最新 ckpt ----
    ckpt_dir = exp_out / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"))
    if not ckpts:
        # checkpoint_every 没触发，强制存最后一个
        last_ckpt = ckpt_dir / f"epoch={num_epochs-1:04d}.ckpt"
        trainer._save_checkpoint(last_ckpt, num_epochs - 1, history["train_loss"][-1])
        print(f"checkpoint_every 未命中，手动保存: {last_ckpt}")
        ckpts = [last_ckpt]
    final_ckpt = ckpts[-1]
    print(f"最新 ckpt: {final_ckpt}")

    # ---- Rollout ----
    if args.skip_rollout:
        print("--skip-rollout 已设置，跳过 rollout")
        return

    use_history = exp.get("use_history", False)
    eval_script = REPO_ROOT / "scripts/evaluation/eval_core_ckpt.py"
    cmd = [
        sys.executable, str(eval_script),
        "--ckpt", str(final_ckpt),
        "--out-dir", str(exp_out),
        "--device", args.device,
        "--n-test", str(args.n_test),
        "--n-test-vis", str(args.n_test_vis),
        "--n-envs", str(args.n_envs),
        "--num-inference-steps", str(args.num_inference_steps),
    ]
    if use_history:
        cmd.append("--use-history")
    print("\n" + "=" * 70)
    print(f"Running rollout: {' '.join(cmd)}")
    print("=" * 70)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"\n✅ {args.exp_id} 完成。结果在 {exp_out}/")


if __name__ == "__main__":
    main()

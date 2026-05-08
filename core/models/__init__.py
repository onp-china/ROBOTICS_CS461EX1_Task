"""core.models: 模型变体（AttnRes、baseline 等）和工具函数。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from resattention_transformer_for_diffusion import ResAttentionTransformerForDiffusion

__all__ = ["build_model", "count_parameters"]


def build_model(variant: str, **kwargs) -> nn.Module:
    """
    按 variant 名称构造模型实例。

    Parameters
    ----------
    variant : str
        - ``'attn_res'`` → ResAttentionTransformerForDiffusion
        - ``'baseline'`` → 标准 TransformerForDiffusion（需要 diffusion_policy）
    **kwargs
        传给模型构造函数的参数，会与下方的 defaults 合并。

    Returns
    -------
    nn.Module
    """
    defaults = dict(
        input_dim=10,
        output_dim=10,
        horizon=16,
        n_obs_steps=2,
        cond_dim=51,
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0,
    )
    cfg = {**defaults, **kwargs}

    if variant == "attn_res":
        return ResAttentionTransformerForDiffusion(
            input_dim=cfg["input_dim"],
            output_dim=cfg["output_dim"],
            horizon=cfg["horizon"],
            n_obs_steps=cfg["n_obs_steps"],
            cond_dim=cfg["cond_dim"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_emb=cfg["n_emb"],
            p_drop_emb=cfg.get("p_drop_emb", 0.0),
            p_drop_attn=cfg["p_drop_attn"],
            causal_attn=cfg["causal_attn"],
            time_as_cond=cfg.get("time_as_cond", True),
            obs_as_cond=cfg.get("obs_as_cond", True),
            n_cond_layers=cfg.get("n_cond_layers", 0),
            attnres_last_bias_init=cfg.get("attnres_last_bias_init", 8.0),
            attnres_temperature=cfg.get("attnres_temperature", 2.0),
            attnres_blend_init=cfg.get("attnres_blend_init", 0.0),
        )

    if variant == "baseline":
        try:
            from diffusion_policy.model.diffusion.transformer_for_diffusion import (
                TransformerForDiffusion,
            )
        except ImportError as exc:
            raise ImportError(
                "baseline 模型需要 diffusion_policy 包（pip install diffusion_policy）"
            ) from exc
        return TransformerForDiffusion(
            input_dim=cfg["input_dim"],
            output_dim=cfg["output_dim"],
            horizon=cfg["horizon"],
            n_obs_steps=cfg["n_obs_steps"],
            cond_dim=cfg["cond_dim"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_emb=cfg["n_emb"],
            p_drop_emb=cfg.get("p_drop_emb", 0.0),
            p_drop_attn=cfg["p_drop_attn"],
            causal_attn=cfg["causal_attn"],
            time_as_cond=cfg.get("time_as_cond", True),
            obs_as_cond=cfg.get("obs_as_cond", True),
            n_cond_layers=cfg.get("n_cond_layers", 0),
        )

    raise ValueError(f"Unknown model variant: {variant!r}")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """返回模型的可训练（或全部）参数量。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

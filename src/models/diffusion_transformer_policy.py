from typing import Dict, Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        device = diffusion_step.device
        half_dim = self.hidden_dim // 2
        factor = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0, device=device)) / max(1, half_dim - 1))
        )
        angles = diffusion_step.float().unsqueeze(1) * factor.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.size(-1) < self.hidden_dim:
            pad = torch.zeros(embedding.size(0), self.hidden_dim - embedding.size(-1), device=device)
            embedding = torch.cat([embedding, pad], dim=-1)
        return embedding


class BaselineTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(attn_out)

        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)
        return x


class AttnResTransformerBlock(nn.Module):
    """
    Minimal switchable AttnRes-style block.

    This keeps the same core attention/FFN path as baseline while adding a
    learnable weighted residual from a running layer summary.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.summary_proj = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        summary: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(attn_out)

        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout(ffn_out)

        gated_alpha = torch.sigmoid(self.alpha)
        x = x + gated_alpha * self.summary_proj(summary)
        next_summary = 0.5 * summary + 0.5 * x
        return x, next_summary


class BaselineDiffusionTransformerPolicy(nn.Module):
    """
    教学版完整 baseline。

    这是当前项目里的默认 baseline 模型，不包含 AttnRes。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_diffusion_step: int,
        residual_mode: str = "standard",
    ) -> None:
        super().__init__()
        if residual_mode not in {"standard", "attnres"}:
            raise ValueError(f"不支持的 residual_mode: {residual_mode}，仅支持 standard / attnres。")

        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        if residual_mode == "attnres":
            self.blocks = nn.ModuleList(
                [AttnResTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [BaselineTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
            )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.max_diffusion_step = max_diffusion_step
        self.residual_mode = residual_mode

    def forward(
        self,
        obs: torch.Tensor,
        noisy_action: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        if obs.ndim != 3 or noisy_action.ndim != 3:
            raise ValueError("obs 和 noisy_action 必须是 [B, T, D] 三维张量。")
        if obs.shape[:2] != noisy_action.shape[:2]:
            raise ValueError("obs 与 noisy_action 的 batch 和序列长度必须一致。")

        sequence_length = obs.shape[1]
        causal_mask = torch.triu(
            torch.ones((sequence_length, sequence_length), device=obs.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.obs_proj(obs) + self.action_proj(noisy_action)
        time_emb = self.time_proj(self.time_embedding(diffusion_step)).unsqueeze(1)
        x = x + time_emb

        if self.residual_mode == "attnres":
            summary = x
            for block in self.blocks:
                x, summary = block(x, summary, attn_mask=causal_mask)
        else:
            for block in self.blocks:
                x = block(x, attn_mask=causal_mask)

        pred_action = self.output_head(x)
        return pred_action, {}

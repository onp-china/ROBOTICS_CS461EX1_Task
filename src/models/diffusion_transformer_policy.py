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
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [BaselineTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.max_diffusion_step = max_diffusion_step

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

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        pred_action = self.output_head(x)
        return pred_action, {}


class AttnResBlock(nn.Module):
    """
    Pre-norm Transformer block with depth-wise learned residual aggregation.

    Each block at depth d learns a softmax-normalized weight vector of length d
    over all previous layer outputs [h_0, ..., h_{d-1}].  The weighted sum
    replaces the plain skip-connection in the attention sub-layer:

        residual_base = Σ_i softmax(α)_i * h_i
        x_new = residual_base + dropout(attn(norm1(h_{d-1})))
        x_new = x_new + dropout(ffn(norm2(x_new)))

    Inspired by Attention Residuals (arXiv:2603.15031).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        depth: int,
    ) -> None:
        """
        depth: 1-indexed position of this block.
               Block 1 aggregates over [h_0] (identity to standard residual at init).
               Block l aggregates over [h_0, ..., h_{l-1}].
        """
        super().__init__()
        self.res_alpha = nn.Parameter(torch.zeros(depth))
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

    def forward(
        self,
        x: torch.Tensor,
        prev_hiddens: list,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # prev_hiddens: [h_0, ..., h_{depth-1}], where h_{depth-1} == x
        weights = torch.softmax(self.res_alpha, dim=0)           # [depth]
        stacked = torch.stack(prev_hiddens, dim=0)               # [depth, B, T, H]
        residual_base = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)  # [B, T, H]

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False, attn_mask=attn_mask)
        x = residual_base + self.dropout(attn_out)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AttnResDiffusionTransformerPolicy(nn.Module):
    """
    扩散 Transformer 策略，注意力子层使用深度方向学习残差聚合（AttnRes）。
    除残差机制外，其余结构与 BaselineDiffusionTransformerPolicy 完全一致，
    方便进行受控对比实验。
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
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [AttnResBlock(hidden_dim, num_heads, dropout, depth=i + 1) for i in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.max_diffusion_step = max_diffusion_step

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

        all_hiddens: list = [x]
        for block in self.blocks:
            x = block(x, all_hiddens, attn_mask=causal_mask)
            all_hiddens.append(x)

        pred_action = self.output_head(x)
        return pred_action, {}

from typing import Dict, List, Optional, Tuple

import math

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
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, residual_mode: str = "block_attnres") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.residual_mode = residual_mode
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
        self.attn_query_scale = nn.Parameter(torch.empty(hidden_dim))
        self.ffn_query_scale = nn.Parameter(torch.empty(hidden_dim))
        self._reset_attnres_parameters()

    def _reset_attnres_parameters(self) -> None:
        nn.init.normal_(self.attn_query_scale, mean=0.0, std=1e-4)
        nn.init.normal_(self.ffn_query_scale, mean=0.0, std=1e-4)

    def _apply_attention_residual(
        self,
        current_state: torch.Tensor,
        branch_output: torch.Tensor,
        history_states: List[torch.Tensor],
        query_scale: torch.Tensor,
    ) -> torch.Tensor:
        candidate_state = current_state + branch_output
        if len(history_states) == 0:
            return candidate_state

        candidate_stack = torch.stack([*history_states, candidate_state], dim=1)

        if self.residual_mode == "uniform_attention":
            weights = candidate_stack.new_full(
                (candidate_stack.size(0), candidate_stack.size(1), candidate_stack.size(2)),
                1.0 / candidate_stack.size(1),
            )
        else:
            query = current_state * query_scale.view(1, 1, -1)
            logits = (candidate_stack * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(candidate_stack.size(-1))
            weights = torch.softmax(logits, dim=1)

        return torch.sum(weights.unsqueeze(-1) * candidate_stack, dim=1)

    def forward(self, x: torch.Tensor, history_states: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if history_states is None:
            history_states = []

        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = self._apply_attention_residual(
            current_state=x,
            branch_output=self.dropout(attn_out),
            history_states=history_states,
            query_scale=self.attn_query_scale,
        )

        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = self._apply_attention_residual(
            current_state=x,
            branch_output=self.dropout(ffn_out),
            history_states=history_states,
            query_scale=self.ffn_query_scale,
        )
        return x


class BaselineDiffusionTransformerPolicy(nn.Module):
    """
    教学版完整 baseline。

    这是当前项目里的默认 AttnRes 增强版 Diffusion Transformer Policy。
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
        attnres_mode: str = "block_attnres",
        attnres_block_size: int = 4,
    ) -> None:
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attnres_block_size = max(1, attnres_block_size)
        self.blocks = nn.ModuleList(
            [BaselineTransformerBlock(hidden_dim, num_heads, dropout, residual_mode=attnres_mode) for _ in range(num_layers)]
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

        x = self.obs_proj(obs) + self.action_proj(noisy_action)
        time_emb = self.time_proj(self.time_embedding(diffusion_step)).unsqueeze(1)
        x = x + time_emb

        hidden_states_history: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, hidden_states_history)
            hidden_states_history.append(x)
            if (layer_idx + 1) % self.attnres_block_size == 0:
                hidden_states_history = []

        pred_action = self.output_head(x)
        return pred_action, {}

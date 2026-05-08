from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class _AttnResDepthMixer(nn.Module):
    """
    Attention Residuals (AttnRes) across depth (layer history).

    We compute a per-sample softmax over a list of hidden states and return their weighted sum.
    The weight logits are produced from a learned query vector q_l (per-layer) dotted with a
    token-averaged summary of each hidden state.

    We add a learned scalar bias to the most recent state to make initialization close to a
    standard residual path (i.e., pick the newest state early in training).
    """

    def __init__(self, d_model: int, last_bias_init: float = 8.0, temperature: float = 2.0) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.zeros(d_model))
        self.last_bias = nn.Parameter(torch.tensor(float(last_bias_init)))
        self.temperature = float(temperature)

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        # states: list of (B,T,D)
        if len(states) == 1:
            return states[0]

        # (B, n, D) token-mean summaries
        summaries = torch.stack([s.mean(dim=1) for s in states], dim=1)
        # (B, n)
        logits = summaries @ self.q
        logits[:, -1] = logits[:, -1] + self.last_bias
        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-6)
        weights = torch.softmax(logits, dim=1)
        # weighted sum on full tensors
        out = 0.0
        for i, s in enumerate(states):
            out = out + weights[:, i].view(-1, 1, 1) * s
        return out


class _ResAttnDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        last_bias_init: float = 8.0,
        attnres_temperature: float = 2.0,
        attnres_blend_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.ffn_drop = nn.Dropout(dropout)

        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        # AttnRes mixers (self-attn residual and FFN residual)
        self.mixer_sa = _AttnResDepthMixer(
            d_model=d_model,
            last_bias_init=last_bias_init,
            temperature=attnres_temperature,
        )
        self.mixer_ff = _AttnResDepthMixer(
            d_model=d_model,
            last_bias_init=last_bias_init,
            temperature=attnres_temperature,
        )

        # Blend AttnRes routing vs standard additive residual output:
        # out = baseline + sigmoid(blend_logit) * (mixed - baseline)
        eps = 1e-6
        b = float(attnres_blend_init)
        b = min(max(b, eps), 1.0 - eps)
        blend_logit = math.log(b / (1.0 - b))
        self.sa_blend_logit = nn.Parameter(torch.tensor(blend_logit, dtype=torch.float32))
        self.ff_blend_logit = nn.Parameter(torch.tensor(blend_logit, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        hist_sa: list[torch.Tensor],
        hist_ff: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        # ---- self-attn block (AttnRes residual) ----
        x_ln = self.norm1(x)
        sa_out = self.self_attn(x_ln, x_ln, x_ln, attn_mask=tgt_mask, need_weights=False)[0]
        x_candidate = x + self.drop1(sa_out)
        mixed = self.mixer_sa(hist_sa + [x_candidate])
        gate = torch.sigmoid(self.sa_blend_logit).to(dtype=mixed.dtype, device=mixed.device)
        x = x_candidate + gate * (mixed - x_candidate)
        hist_sa.append(x)

        # ---- cross-attn block (standard residual) ----
        x_ln = self.norm2(x)
        ca_out = self.cross_attn(
            x_ln, memory, memory, attn_mask=memory_mask, need_weights=False
        )[0]
        x = x + self.drop2(ca_out)

        # ---- FFN block (AttnRes residual) ----
        x_ln = self.norm3(x)
        ff = self.linear2(self.ffn_drop(self.act(self.linear1(x_ln))))
        x_candidate = x + self.drop3(ff)
        mixed = self.mixer_ff(hist_ff + [x_candidate])
        gate = torch.sigmoid(self.ff_blend_logit).to(dtype=mixed.dtype, device=mixed.device)
        x = x_candidate + gate * (mixed - x_candidate)
        hist_ff.append(x)

        return x, hist_sa, hist_ff


class _ResAttnTransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        last_bias_init: float = 8.0,
        attnres_temperature: float = 2.0,
        attnres_blend_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ResAttnDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    last_bias_init=last_bias_init,
                    attnres_temperature=attnres_temperature,
                    attnres_blend_init=attnres_blend_init,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        hist_sa: list[torch.Tensor] = []
        hist_ff: list[torch.Tensor] = []
        for layer in self.layers:
            x, hist_sa, hist_ff = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                hist_sa=hist_sa,
                hist_ff=hist_ff,
            )
        return x


class ResAttentionTransformerForDiffusion(ModuleAttrMixin):
    """
    Diffusion Transformer denoising backbone with Attention Residuals (AttnRes) across depth.

    This is a drop-in replacement for
    `diffusion_policy.model.diffusion.transformer_for_diffusion.TransformerForDiffusion`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int | None = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        attnres_last_bias_init: float = 8.0,
        attnres_temperature: float = 2.0,
        attnres_blend_init: float = 0.0,
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb),
                )

            self.decoder = _ResAttnTransformerDecoder(
                n_layer=n_layer,
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                last_bias_init=attnres_last_bias_init,
                attnres_temperature=attnres_temperature,
                attnres_blend_init=attnres_blend_init,
            )
        else:
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

        # attention mask
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
                mask = t >= (s - 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.GELU,
            nn.TransformerEncoderLayer,
            nn.TransformerEncoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            _ResAttnTransformerDecoder,
            _ResAttnDecoderLayer,
            _AttnResDepthMixer,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, ResAttentionTransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                # explicit non-decay params
                if pn in ("q", "last_bias", "sa_blend_logit", "ff_blend_logit"):
                    no_decay.add(fpn)
                elif pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        # Any remaining parameters (e.g. AttnRes query vectors) default to no_decay.
        for pn in (param_dict.keys() - union_params):
            no_decay.add(pn)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x

            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)
        return x


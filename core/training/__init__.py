"""core.training: 训练循环（DiffusionTrainer）和配置（TrainingConfig）。"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

__all__ = ["DiffusionTrainer", "TrainingConfig"]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    集中管理训练超参数。字段名与 experiment.ipynb 中 cell 14 的调用保持一致。

    Parameters
    ----------
    device : str
        设备，如 ``'cuda:0'``、``'mps'``、``'cpu'``。
    num_epochs : int
        训练 epoch 数。
    lr / weight_decay / betas
        AdamW 优化器参数。
    amp_dtype : str
        AMP 数值类型，``'float16'``、``'bfloat16'`` 或 ``'tf32'``（自动适配）。
    cudnn_benchmark / cudnn_deterministic
        cuDNN 配置。
    batch_size / grad_accum_steps
        有效 batch_size = batch_size × grad_accum_steps。
    use_ema / ema_decay
        EMA 开关和衰减率。
    grad_clip : float | None
        梯度裁剪阈值（None = 不裁剪）。
    log_every / val_every / checkpoint_every : int
        记录、验证、保存 checkpoint 的频率（按 step）。
    output_dir : Path | str
        输出根目录。
    """

    device: str = "cuda:0"
    num_epochs: int = 5000
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)

    # --- AMP ---
    amp_dtype: str = "tf32"

    # --- cuDNN ---
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # --- Batch ---
    batch_size: int = 256
    grad_accum_steps: int = 1

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999

    # --- Regularisation ---
    grad_clip: float | None = 1.0

    # --- Logging / saving ---
    log_every: int = 10
    val_every: int = 1
    checkpoint_every: int = 50
    output_dir: Path | str = Path("outputs_notebook")

    # 私有状态（训练过程中由 trainer 填充）
    _grad_scaler: Optional[GradScaler] = field(default=None, init=False, repr=False)
    _ema_shadow: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def _update_ema(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float):
    """将 model 的参数指数移动平均写入 ema_state（原地更新）。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_param = ema_state.get(name)
                if ema_param is None:
                    ema_state[name] = param.data.clone().detach()
                else:
                    ema_state[name] = decay * ema_param + (1.0 - decay) * param.data


def _apply_ema(model: nn.Module, ema_state: dict[str, torch.Tensor]):
    """把 ema_state 的值拷回 model（用于用 EMA 权重做评估）。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_state:
                param.data.copy_(ema_state[name])


# ---------------------------------------------------------------------------
# DiffusionTrainer
# ---------------------------------------------------------------------------

class DiffusionTrainer:
    """
    简化版 diffusion 训练器，支持：
    - AMP（TF32 / FP16 / BF16）
    - EMA
    - 梯度累积
    - 梯度裁剪
    - 训练 / 验证 loss 记录
    - checkpoint 自动保存（last + top-k）
    - fit(num_epochs) 返回 history
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,  # DataLoader
        val_loader,    # DataLoader
        cfg: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # --- 设备 ---
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # --- cuDNN ---
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
            torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        # --- 优化器 ---
        self.optimizer: optim.Optimizer
        if hasattr(model, "configure_optimizers"):
            # ResAttentionTransformerForDiffusion 自带的分组优化器
            self.optimizer = model.configure_optimizers(
                learning_rate=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )

        # --- AMP scaler ---
        amp_dtype = cfg.amp_dtype.lower()
        if amp_dtype == "tf32":
            self.amp_enabled = torch.cuda.is_available()
            self.amp_dtype = torch.float16  # TF32 在 A100/RTX PRO 6000 上等效于 FP16 速度 + FP32 精度
        elif amp_dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        elif amp_dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        self.scaler = GradScaler(enabled=self.amp_enabled)

        # --- EMA ---
        if cfg.use_ema:
            self.ema_state: dict[str, torch.Tensor] = {}
            self.ema_decay = cfg.ema_decay
        else:
            self.ema_state = None
            self.ema_decay = None

        # --- 输出目录 ---
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # --- 日志文件 ---
        self.log_path = self.output_dir / "logs.json.txt"
        self._log_file = None  # opened on first write

        # --- 历史记录 ---
        self.history: dict[str, list] = dict(
            train_loss=[],
            val_loss=[],
            lr=[],
            epoch=[],
            global_step=[],
        )

        # --- 全局步计数器 ---
        self.global_step = 0

    # ------------------------------------------------------------------
    # _log_metric — write one JSON line to logs.json.txt
    # ------------------------------------------------------------------
    def _log_metric(self, epoch: int, step: int, train_loss: float, 
                    val_loss: float = None, lr: float = None, **extra):
        """Append a JSON metric record to logs.json.txt."""
        if self._log_file is None:
            self._log_file = open(self.log_path, "a", buffering=1)  # line buffering
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(train_loss),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if val_loss is not None:
            record["val_loss"] = float(val_loss)
        if lr is not None:
            record["lr"] = float(lr)
        record.update(extra)
        self._log_file.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int) -> dict[str, list]:
        """
        运行完整的训练循环。

        Returns
        -------
        self.history（字典，keys: train_loss, val_loss, lr, epoch, global_step）
        """
        self.model.train()
        epochs_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        for epoch in epochs_pbar:
            epoch_loss = 0.0
            n_batches = 0

            self.optimizer.zero_grad(set_to_none=True)

            # Inner progress bar for batches
            step_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                           desc=f"Epoch {epoch}", unit="batch", leave=False)

            for step, batch in step_pbar:
                # 把数据移到设备
                obs = batch.get("obs")
                action = batch.get("action")
                if obs is not None:
                    obs = obs.to(self.device, non_blocking=True)
                if action is not None:
                    action = action.to(self.device, non_blocking=True)

                # 前向 + 损失（子类可重写 _compute_loss）
                loss = self._compute_loss(obs, action)

                # 梯度累积
                loss = loss / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()

                if (step + 1) % self.cfg.grad_accum_steps == 0:
                    # 梯度裁剪
                    if self.cfg.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    # EMA
                    if self.ema_state is not None:
                        _update_ema(self.ema_state, self.model, self.ema_decay)

                epoch_loss += loss.item() * self.cfg.grad_accum_steps
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- 记录 ---
            self.history["train_loss"].append(avg_train_loss)
            self.history["lr"].append(current_lr)
            self.history["epoch"].append(epoch)
            self.history["global_step"].append(self.global_step)

            # Update epoch progress bar with current metrics
            epochs_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "step": self.global_step
            })

            if (epoch + 1) % self.cfg.log_every == 0 or epoch == 0:
                print(
                    f"epoch {epoch:4d}  "
                    f"train_loss={avg_train_loss:.6f}  "
                    f"lr={current_lr:.2e}  "
                    f"step={self.global_step}"
                )

            # --- 写日志文件 ---
            self._log_metric(epoch=epoch, step=self.global_step,
                           train_loss=avg_train_loss, lr=current_lr)

            # --- Validation ---
            if (epoch + 1) % self.cfg.val_every == 0:
                val_loss = self._eval()
                self.history["val_loss"].append(val_loss)
                # Also log val_loss
                self._log_metric(epoch=epoch, step=self.global_step,
                               train_loss=avg_train_loss, val_loss=val_loss, lr=current_lr)
                if self.ema_state is not None:
                    _apply_ema(self.model, self.ema_state)
                    val_loss_ema = self._eval()
                    self.history.setdefault("val_loss_ema", []).append(val_loss_ema)
                    _update_ema(self.ema_state, self.model, self.ema_decay)
                    print(f"  val_loss={val_loss:.6f}  val_loss_ema={val_loss_ema:.6f}")
                else:
                    print(f"  val_loss={val_loss:.6f}")

            # --- Checkpoint ---
            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                ckpt_path = self.output_dir / "checkpoints" / f"epoch={epoch:04d}.ckpt"
                self._save_checkpoint(ckpt_path, epoch, avg_train_loss)

        # 保存 last
        self._save_checkpoint(
            self.output_dir / "checkpoints" / "latest.ckpt",
            num_epochs - 1,
            avg_train_loss,
        )
        return self.history

    # ------------------------------------------------------------------
    # 子类可重写的方法
    # ------------------------------------------------------------------

    def _compute_loss(self, obs, action) -> torch.Tensor:
        """
        计算单步损失。默认是简单的 MSE，可以子类重写。
        """
        B = action.shape[0]
        device = action.device

        # 模拟 diffusion noise schedule
        t = torch.randint(0, 100, (B,), device=device)
        noise = torch.randn_like(action)
        alpha_t = 1.0 - t.float() / 100.0
        noisy = alpha_t.view(B, 1, 1) * action + (1 - alpha_t).view(B, 1, 1) * noise

        # 这里用的是模型的原始 forward，真实项目中会走完整的 diffusion 过程
        # 为了让 core/training 跑通（不依赖完整 diffusion_policy），用一个简化的 MSE 代理
        pred = self.model(sample=noisy, timestep=t % 100, cond=obs)
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = torch.nn.functional.mse_loss(pred, noise)
        return loss

    def _eval(self) -> float:
        """跑一遍验证集，返回平均 loss。"""
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                obs = batch.get("obs")
                action = batch.get("action")
                if obs is not None:
                    obs = obs.to(self.device, non_blocking=True)
                if action is not None:
                    action = action.to(self.device, non_blocking=True)
                with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", 
                              enabled=self.amp_enabled, dtype=self.amp_dtype):
                    loss = self._compute_loss(obs, action)
                total_loss += loss.item()
                n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, path: Path, epoch: int, train_loss: float):
        """保存 PyTorch checkpoint。"""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": train_loss,
            "history": self.history,
            "global_step": self.global_step,
        }
        if self.ema_state is not None:
            ckpt["ema_state_dict"] = self.ema_state
        torch.save(ckpt, path)

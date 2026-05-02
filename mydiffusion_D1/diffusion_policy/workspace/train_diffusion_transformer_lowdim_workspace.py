if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import sys
from torch.utils.data import DataLoader
import copy
import random
import wandb
import numpy as np
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

MYDIFFUSION_D1_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(MYDIFFUSION_D1_ROOT) not in sys.path:
    sys.path.insert(0, str(MYDIFFUSION_D1_ROOT))

from _runtime import read_json_lines
from scripts.export_rollout_video import export_rollout_video_for_checkpoint
from scripts.plot_results import plot_single_run_curves_for_dir

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def _resolve_best_checkpoint_path(self) -> pathlib.Path | None:
        checkpoint_dir = pathlib.Path(self.output_dir) / "checkpoints"
        if not checkpoint_dir.is_dir():
            return None

        rollout_checkpoint = self._resolve_best_rollout_checkpoint_path(checkpoint_dir)
        if rollout_checkpoint is not None:
            return rollout_checkpoint

        monitor_key = str(self.cfg.checkpoint.topk.monitor_key)
        mode = str(self.cfg.checkpoint.topk.mode)
        candidates = []
        for ckpt_path in checkpoint_dir.glob("*.ckpt"):
            if ckpt_path.name == "latest.ckpt":
                continue
            stem = ckpt_path.stem
            marker = f"{monitor_key}="
            if marker not in stem:
                continue
            try:
                raw_value = stem.split(marker, 1)[1].split("-", 1)[0]
                metric_value = float(raw_value)
            except ValueError:
                continue
            candidates.append((metric_value, ckpt_path))

        if not candidates:
            latest_path = checkpoint_dir / "latest.ckpt"
            return latest_path if latest_path.is_file() else None

        if mode == "max":
            return max(candidates, key=lambda item: item[0])[1]
        return min(candidates, key=lambda item: item[0])[1]

    def _resolve_best_rollout_checkpoint_path(self, checkpoint_dir: pathlib.Path) -> pathlib.Path | None:
        log_path = pathlib.Path(self.output_dir) / "logs.json.txt"
        rows = read_json_lines(log_path)
        if not rows:
            return None

        rollout_rows = [row for row in rows if "epoch" in row and any(key in row for key in (
            "test/contact_rate",
            "test/mean_min_eef_object_distance",
            "test/mean_object_displacement",
            "test/mean_score",
        ))]
        if not rollout_rows:
            return None

        best_row = max(
            rollout_rows,
            key=lambda row: (
                float(row.get("test/contact_rate", 0.0)),
                -float(row.get("test/mean_min_eef_object_distance", float("inf"))),
                float(row.get("test/mean_object_displacement", 0.0)),
                float(row.get("test/mean_score", 0.0)),
                -float(row.get("val_loss", float("inf"))),
                int(row.get("epoch", -1)),
            ),
        )
        epoch = int(best_row.get("epoch", -1))
        if epoch < 0:
            return None

        for ckpt_path in checkpoint_dir.glob(f"epoch={epoch:04d}-*.ckpt"):
            return ckpt_path
        return None

    def _maybe_export_best_rollout_video(self) -> None:
        if not bool(getattr(self.cfg.training, "auto_export_best_rollout_video", False)):
            return

        best_checkpoint = self._resolve_best_checkpoint_path()
        if best_checkpoint is None:
            print("Skipping best-checkpoint rollout export: no checkpoint found.")
            return

        output_dir = pathlib.Path(self.output_dir) / "best_rollout_export"
        output_video = pathlib.Path(self.output_dir) / "best_epoch_rollout.mp4"
        try:
            summary = export_rollout_video_for_checkpoint(
                task=str(self.cfg.task.name),
                checkpoint=best_checkpoint,
                output_dir=output_dir,
                output_video=output_video,
                seed=int(getattr(self.cfg.training, "best_rollout_seed", 100000)),
                n_test=int(getattr(self.cfg.training, "best_rollout_n_test", 1)),
                n_envs=int(getattr(self.cfg.training, "best_rollout_n_envs", 1)),
                device=str(self.cfg.training.device),
            )
            print(
                "Best-checkpoint rollout video exported: "
                f"checkpoint={best_checkpoint} video={summary.get('video_path')} score={summary.get('mean_score')}"
            )
        except Exception as exc:
            print(
                "Best-checkpoint rollout export failed: "
                f"checkpoint={best_checkpoint} error={exc}"
            )

    def _maybe_export_training_curves(self) -> None:
        if not bool(getattr(self.cfg.training, "auto_export_training_curves", False)):
            return

        run_dir = pathlib.Path(self.output_dir)
        curves_dir = run_dir / "curves"
        try:
            saved = plot_single_run_curves_for_dir(
                run_dir=run_dir,
                task=str(self.cfg.task.name),
                seed=int(self.cfg.training.seed),
                curves_dir=curves_dir,
            )
            if saved:
                print("Training curves exported: " + ", ".join(str(path) for path in saved))
            else:
                print("Skipping training curve export: no plottable metrics found.")
        except Exception as exc:
            print(f"Training curve export failed: {exc}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        env_runner = None

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        if (not cfg.training.resume) and os.path.isfile(log_path):
            # Start a fresh json log for non-resumed runs instead of appending history.
            os.remove(log_path)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                for batch_idx, batch in enumerate(train_dataloader):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    train_losses.append(raw_loss.item())

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log = {
                    'train_loss': train_loss,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if bool(getattr(cfg.training, "enable_rollout", False)) and cfg.training.rollout_every > 0 and (
                    self.epoch % cfg.training.rollout_every
                ) == 0:
                    if env_runner is None:
                        env_runner = hydra.utils.instantiate(
                            cfg.task.env_runner,
                            output_dir=self.output_dir)
                        assert isinstance(env_runner, BaseLowdimRunner)
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_mse_values = list()
                        for batch_idx, batch in enumerate(val_dataloader):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            obs_dict = {'obs': batch['obs']}
                            gt_action = batch['action']
                            result = policy.predict_action(obs_dict)
                            if cfg.pred_action_steps_only:
                                pred_action = result['action']
                                start = cfg.n_obs_steps - 1
                                end = start + cfg.n_action_steps
                                gt_action = gt_action[:,start:end]
                            else:
                                pred_action = result['action_pred']
                            val_mse_values.append(torch.nn.functional.mse_loss(pred_action, gt_action).item())
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                        if len(val_mse_values) > 0:
                            step_log['val_action_mse_error'] = float(np.mean(val_mse_values))
            
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                print(
                    f"[Epoch {self.epoch}] "
                    f"train_loss={step_log['train_loss']:.6f} "
                    + (
                        f"val_loss={step_log['val_loss']:.6f} "
                        if 'val_loss' in step_log else ""
                    )
                    + (
                        f"train_action_mse_error={step_log['train_action_mse_error']:.6f} "
                        if 'train_action_mse_error' in step_log else ""
                    )
                    + (
                        f"test_mean_score={step_log['test/mean_score']:.6f} "
                        if 'test/mean_score' in step_log else ""
                    )
                    + (
                        f"test_contact_rate={step_log['test/contact_rate']:.6f} "
                        if 'test/contact_rate' in step_log else ""
                    )
                    + (
                        f"test_mean_min_eef_object_distance={step_log['test/mean_min_eef_object_distance']:.6f} "
                        if 'test/mean_min_eef_object_distance' in step_log else ""
                    )
                    + (
                        f"test_mean_object_displacement={step_log['test/mean_object_displacement']:.6f} "
                        if 'test/mean_object_displacement' in step_log else ""
                    )
                    + f"lr={step_log['lr']:.6e}"
                )
                self.global_step += 1
                self.epoch += 1
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        self.wait_for_saves()
        self._maybe_export_training_curves()
        self._maybe_export_best_rollout_video()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

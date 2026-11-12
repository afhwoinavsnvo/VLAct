# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args
from starVLA.training.trainer_utils.wandb_setup import resolve_wandb_entity

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def create_accelerator(cfg) -> Accelerator:
    gradient_accumulation_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    deepspeed_plugin = (None if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1"
                       else DeepSpeedPlugin(gradient_accumulation_steps=gradient_accumulation_steps))
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    accelerator.print(accelerator.state)
    return accelerator


def _unwrap_model(accelerator, model):
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        return model.module if hasattr(model, "module") else model
    return accelerator.unwrap_model(model)


def _get_state_dict(accelerator, model):
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        return _unwrap_model(accelerator, model).state_dict()
    return accelerator.get_state_dict(model)


def _as_bool(value) -> bool:
    """Parse optional CLI/config booleans safely."""
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


def initialize_wandb(config, accelerator):
    """Initialize the actual run before loading model weights on any rank."""
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        return False
    failure = [None]
    local_error = None
    if accelerator.is_main_process:
        try:
            log_dir = Path(config.output_dir) / "wandb"
            log_dir.mkdir(parents=True, exist_ok=True)
            full_config = config.unwrap() if isinstance(config, AccessTrackedConfig) else config
            full_config = (OmegaConf.to_container(full_config, resolve=True)
                           if OmegaConf.is_config(full_config) else dict(full_config))
            run = wandb.init(
                name=os.environ.get("WANDB_NAME") or config.run_id,
                dir=str(log_dir),
                project=os.environ.get("WANDB_PROJECT") or config.wandb_project,
                entity=resolve_wandb_entity(config),
                group="vla-train",
                config=full_config,
            )
            if getattr(run, "url", None):
                logger.info(f"W&B training dashboard: {run.url}")
        except Exception as error:
            local_error = error
            failure[0] = f"W&B initialization failed before model loading: {error}"
    if dist.is_initialized():
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(failure[0]) from local_error
    accelerator.wait_for_everyone()
    return True


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        if not hasattr(self, "_wandb_enabled"):
            self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Compatibility path for callers constructing a trainer directly."""
        self._wandb_enabled = initialize_wandb(self.config, self.accelerator)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        reset_steps = _as_bool(getattr(self.config.trainer, "reset_steps", False))
        random_init_action_model = _as_bool(getattr(self.config.trainer, "random_init_action_model", False))
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, resumed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self.completed_steps = 0 if reset_steps else resumed_steps
                if reset_steps:
                    logger.info(
                        f"Loaded checkpoint {self.resume_from_checkpoint} and reset completed_steps to 0."
                    )
                else:
                    logger.info(
                        f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            ignore_modules = None
            if random_init_action_model:
                logger.info("`random_init_action_model=True`, skipping `action_model` when loading checkpoint.")
                ignore_modules = "action_model"
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                ignore_modules=ignore_modules,
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            if reset_steps:
                logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, reset completed_steps to 0.")
            else:
                logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = _get_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            rates = self.lr_scheduler.get_last_lr()
            metrics["learning_rate"] = rates[0]
            for index, group in enumerate(self.optimizer.param_groups):
                metrics[f"learning_rate/{group.get('name', index)}"] = rates[index]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                wandb.log(metrics, step=self.completed_steps, commit=True)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        # from_pretrained 默认 eval 模式；不转 train 的话 gradient checkpointing 不生效
        self.model.train()
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.accelerator.sync_gradients:
                eval_interval = int(self.config.trainer.get("eval_interval", 0))
                if (eval_interval > 0 and self.completed_steps % eval_interval == 0
                        and self._action_error_diagnostics_enabled()):
                    step_metrics = self.eval_action_model(step_metrics)

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _action_error_diagnostics_enabled(self) -> bool:
        """Raw subtraction does not match OFT's wrapped-angle objective."""
        framework = self.config.get("framework", {})
        data = self.config.get("datasets", {}).get("vla_data", {})
        wrapped_oft = framework.get("name") == "QwenOFT" and (
            "wrap" in str(data.get("data_mix", ""))
            or _as_bool(self.config.trainer.get("shortest_angular_joint_loss", False))
        )
        return not wrapped_oft

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Optional raw-action diagnostics for non-wrapped representations."""
        step_metrics = {} if step_metrics is None else step_metrics
        if not self._action_error_diagnostics_enabled():
            return step_metrics
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = _unwrap_model(self.accelerator, self.model).predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            pred_actions, gt_actions, valid_mask = self._align_eval_actions(
                predicted=normalized_actions,
                ground_truth=actions,
            )
            valid_count = np.count_nonzero(valid_mask)
            if valid_count == 0:
                raise ValueError("Cannot measure action error without valid target cells")
            difference = pred_actions.astype(np.float64) - gt_actions.astype(np.float64)
            # Training-loader diagnostics, not held-out validation or success rate.
            step_metrics["train_action_mse"] = float(np.sum(difference ** 2 * valid_mask) / valid_count)
            step_metrics["train_action_mae"] = float(np.sum(np.abs(difference) * valid_mask) / valid_count)

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    @staticmethod
    def _align_eval_actions(predicted, ground_truth):
        """Align heterogeneous action tensors for evaluation.

        Supports mixed-embodiment batches where action dims may vary across samples.
        Valid dimensions are derived from the ground-truth action shape, then both
        prediction and target are padded/truncated to a shared max dim for masked
        distance computation.
        """
        gt_list = [np.asarray(action, dtype=np.float32) for action in ground_truth]

        if isinstance(predicted, np.ndarray):
            if predicted.ndim == 2:
                pred_list = [np.asarray(predicted, dtype=np.float32)]
            elif predicted.ndim == 3:
                pred_list = [np.asarray(predicted[idx], dtype=np.float32) for idx in range(predicted.shape[0])]
            else:
                raise ValueError(f"Unsupported predicted action shape: {predicted.shape}")
        elif isinstance(predicted, (list, tuple)):
            pred_list = [np.asarray(action, dtype=np.float32) for action in predicted]
        else:
            raise TypeError(f"Unsupported predicted action type: {type(predicted).__name__}")

        if len(pred_list) != len(gt_list):
            raise ValueError(
                f"Prediction/ground-truth batch size mismatch: {len(pred_list)} vs {len(gt_list)}"
            )

        max_horizon = max(max(pred.shape[0], gt.shape[0]) for pred, gt in zip(pred_list, gt_list))
        max_dim = max(max(pred.shape[-1], gt.shape[-1]) for pred, gt in zip(pred_list, gt_list))

        pred_batch = np.zeros((len(pred_list), max_horizon, max_dim), dtype=np.float32)
        gt_batch = np.zeros((len(gt_list), max_horizon, max_dim), dtype=np.float32)
        valid_mask = np.zeros((len(gt_list), max_horizon, max_dim), dtype=np.float32)

        for idx, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            pred_horizon = min(pred.shape[0], max_horizon)
            pred_dim = min(pred.shape[-1], max_dim)
            gt_horizon = min(gt.shape[0], max_horizon)
            gt_dim = min(gt.shape[-1], max_dim)

            pred_batch[idx, :pred_horizon, :pred_dim] = pred[:pred_horizon, :pred_dim]
            gt_batch[idx, :gt_horizon, :gt_dim] = gt[:gt_horizon, :gt_dim]
            valid_mask[idx, :gt_horizon, :gt_dim] = 1.0

        return pred_batch, gt_batch, valid_mask

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        metrics = {
            "action_loss": action_loss.item(),
            "action_dit_loss": action_loss.item(),
        }
        if "action_dit_loss" in output_dict:
            metrics["action_dit_loss"] = output_dict["action_dit_loss"].item()
        if "oft_loss" in output_dict:
            metrics["oft_loss"] = output_dict["oft_loss"].item()
        if "gr00t_loss" in output_dict:
            metrics["gr00t_loss"] = output_dict["gr00t_loss"].item()
        if "pi_loss" in output_dict:
            metrics["pi_loss"] = output_dict["pi_loss"].item()
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = _get_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg, accelerator) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    wandb_enabled = initialize_wandb(cfg, accelerator)
    # Initialize fresh heads/adapters reproducibly before rank-specific seeds.
    set_seed(int(cfg.seed))
    vla = build_framework(cfg)
    if accelerator.is_main_process:
        full_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        OmegaConf.save(full_cfg, output_dir / "config.yaml", resolve=True)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer._wandb_enabled = wandb_enabled
    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    accelerator = create_accelerator(cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    try:
        main(cfg, accelerator)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

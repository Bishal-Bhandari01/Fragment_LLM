"""
Trainer for Fragment_LLM — production-ready, DDP-aware.

Features
--------
  - Multi-GPU via DistributedDataParallel (launched with torchrun)
  - bf16 / fp16 mixed precision (GradScaler only for fp16)
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpointing every N steps + on-epoch
  - Resume from any checkpoint
  - W&B logging (optional)
  - Rank-0 only logging/saving
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def _is_main() -> bool:
    """True for rank 0 (or non-DDP runs)."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _lr_cosine_warmup(step: int, warmup: int, max_iters: int,
                      lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    import math
    if step < warmup:
        return lr * step / max(1, warmup)
    if step >= max_iters:
        return min_lr
    progress = (step - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """
    Full training loop with:
      - Gradient accumulation
      - Mixed precision (bf16 or fp16)
      - DDP (optional, transparent)
      - LR warmup + cosine decay
      - Automatic checkpoint / resume
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_loader,
        config,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        save_every_n_steps: int = 1_000,
    ) -> None:
        self.config    = config
        self.device    = torch.device(device)
        self.ckpt_dir  = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every_n_steps

        # ── Move model ──────────────────────────────────────────────────
        self.model = model.to(self.device)

        # ── DDP wrap ────────────────────────────────────────────────────
        self.ddp = dist.is_available() and dist.is_initialized()
        if self.ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(self.model, device_ids=[local_rank])

        self.raw_model = self.model.module if self.ddp else self.model

        # ── Data ────────────────────────────────────────────────────────
        self.train_loader = train_dataloader
        self.val_loader   = val_loader

        # ── Precision ───────────────────────────────────────────────────
        self.use_bf16  = config.use_bf16 and device != "cpu"
        self.use_amp   = config.use_amp  and device != "cpu"
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        # GradScaler only needed for fp16 (bf16 is numerically stable)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and not self.use_bf16))

        # ── Optimiser ───────────────────────────────────────────────────
        # Weight decay only on 2-D parameters (weight matrices), not biases / norms
        decay_params   = [p for n, p in self.raw_model.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.raw_model.named_parameters()
                          if p.requires_grad and p.dim() < 2]

        self.optimizer = AdamW(
            [
                {"params": decay_params,   "weight_decay": config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),  # faster fused kernel on GPU
        )

        # ── Training state ───────────────────────────────────────────────
        self.global_step = 0
        self.start_epoch = 0

        if _is_main():
            n = sum(p.numel() for p in self.raw_model.parameters())
            logger.info(f"Trainer ready  |  params={n:,}  |  device={device}  |  "
                        f"ddp={self.ddp}  |  bf16={self.use_bf16}")

    # ── LR scheduling ─────────────────────────────────────────────────────────

    def _set_lr(self, step: int) -> float:
        cfg = self.config
        lr = _lr_cosine_warmup(
            step,
            getattr(cfg, "warmup_iters", 0),
            cfg.max_iters,
            cfg.learning_rate,
            getattr(cfg, "min_lr", cfg.learning_rate / 10),
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str) -> None:
        if not _is_main():
            return
        path = self.ckpt_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "model":     self.raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler":    self.scaler.state_dict(),
                "step":      self.global_step,
                "config":    self.config,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt.get("step", 0)
        logger.info(f"Resumed from {path}  (step={self.global_step})")

    # ── Single training step ──────────────────────────────────────────────────

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            _, loss = self.model(x, y)
            loss = loss / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        return loss.item() * self.config.gradient_accumulation_steps

    # ── Epoch loop ────────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        # DDP sampler needs to know the epoch for proper shuffling
        if self.ddp and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_batches  = 0
        accum      = self.config.gradient_accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not _is_main())

        for step_idx, (x, y) in enumerate(pbar):
            # LR schedule
            lr = self._set_lr(self.global_step)

            loss_val = self._train_step(x, y)

            # Gradient accumulation: only step every *accum* micro-batches
            if (step_idx + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.raw_model.parameters(),
                                         self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Periodic checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step{self.global_step:08d}")

                if _is_main():
                    pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr:.2e}"})
                    if _WANDB and wandb.run:
                        wandb.log({"train/loss": loss_val, "train/lr": lr,
                                   "step": self.global_step})

            total_loss += loss_val
            n_batches  += 1

            if self.global_step >= self.config.max_iters:
                break

        return total_loss / max(1, n_batches)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        for x, y in tqdm(self.val_loader, desc="Validation", disable=not _is_main()):
            x, y = x.to(self.device), y.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                _, loss = self.model(x, y)
            total_loss += loss.item()
            n += 1

        avg = total_loss / max(1, n)

        if _is_main():
            if _WANDB and wandb.run:
                wandb.log({"val/loss": avg, "step": self.global_step})

        return avg

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self, epochs: int) -> None:
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate()
            elapsed    = time.time() - t0

            if _is_main():
                logger.info(
                    f"Epoch {epoch+1}/{epochs}  "
                    f"train={train_loss:.4f}  val={val_loss:.4f}  "
                    f"time={elapsed:.1f}s  step={self.global_step}"
                )
                self.save_checkpoint(f"epoch{epoch+1:03d}")

            if self.global_step >= self.config.max_iters:
                logger.info("Reached max_iters — stopping.")
                break
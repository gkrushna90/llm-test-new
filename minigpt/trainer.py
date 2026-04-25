"""
Trainer — Handles training loop, checkpointing, logging, and evaluation.
Supports gradient accumulation, LR scheduling, mixed precision, and optional wandb.
"""

import os
import csv
import time
import math
import torch
from torch.amp import GradScaler, autocast

from minigpt.config import GPTConfig
from minigpt.model import GPT
from minigpt.tokenizer import Tokenizer


class Trainer:
    """
    Handles the full training pipeline.

    Usage:
        trainer = Trainer(config, model, tokenizer, train_loader, val_loader)
        trainer.train()
    """

    def __init__(self, config: GPTConfig, model: GPT, tokenizer: Tokenizer,
                 train_loader, val_loader=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision
        self.use_amp = config.device == "cuda" and config.dtype in ("float16", "bfloat16")
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.amp_dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16

        # State
        self.start_iter = 0
        self.best_val_loss = float('inf')

        # Ensure output dir exists
        os.makedirs(config.output_dir, exist_ok=True)

        # wandb
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name or None,
                    config=vars(config),
                )
            except ImportError:
                print("WARNING: wandb not installed. Skipping wandb logging.")

    def _get_lr(self, it: int) -> float:
        """Cosine learning rate schedule with warmup."""
        cfg = self.config
        # Warmup
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / cfg.warmup_iters
        # Cosine decay
        decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """Estimate loss on train and val splits."""
        self.model.eval()
        losses = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            if loader is None:
                continue
            total_loss = 0.0
            count = 0
            for i, (x, y) in enumerate(loader):
                if i >= self.config.eval_iters:
                    break
                x, y = x.to(self.config.device), y.to(self.config.device)
                _, loss = self.model(x, y)
                total_loss += loss.item()
                count += 1
            if count > 0:
                losses[split] = total_loss / count
        self.model.train()
        return losses

    def save_checkpoint(self, step: int, loss: float):
        """Save training checkpoint with model, optimizer, tokenizer, and config."""
        checkpoint = {
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
            "config": vars(self.config),
            # Tokenizer data for standalone inference
            "stoi": self.tokenizer.stoi,
            "itos": self.tokenizer.itos,
            "vocab_size": self.tokenizer.vocab_size,
            "tokenizer_backend": self.tokenizer.backend,
        }
        torch.save(checkpoint, self.config.checkpoint_path)

    def load_checkpoint(self):
        """Resume from checkpoint if it exists."""
        path = self.config.checkpoint_path
        if not os.path.exists(path):
            print("No checkpoint found. Starting fresh.")
            return

        checkpoint = torch.load(path, map_location=self.config.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_iter = checkpoint["step"] + 1
            print(f"Resumed from checkpoint at step {checkpoint['step']} (loss {checkpoint['loss']:.4f})")
        except RuntimeError:
            print("⚠ Checkpoint architecture mismatch — starting fresh training.")
            print(f"  (Delete {path} to hide this warning.)")

    def train(self):
        """Main training loop."""
        cfg = self.config
        model = self.model
        model.train()

        # Resume
        self.load_checkpoint()

        # Prepare CSV log
        log_exists = os.path.exists(cfg.log_path)
        log_file = open(cfg.log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        if not log_exists:
            log_writer.writerow(["step", "train_loss", "val_loss", "lr", "time_sec"])

        print(f"\nTraining for {cfg.max_iters} steps on {cfg.device}")
        print(f"Model: {model.count_parameters():,} parameters")
        print("-" * 60)

        train_iter = iter(self.train_loader)
        t0 = time.time()

        for step in range(self.start_iter, cfg.max_iters):
            # Update learning rate
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Get batch (handle DataLoader exhaustion)
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                xb, yb = next(train_iter)

            xb, yb = xb.to(cfg.device), yb.to(cfg.device)

            # Forward + backward with optional mixed precision
            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype):
                    _, loss = model(xb, yb)
                    loss = loss / cfg.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                _, loss = model(xb, yb)
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.grad_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Logging & checkpointing
            if step % cfg.eval_interval == 0:
                dt = time.time() - t0
                losses = self.estimate_loss()
                train_loss = losses.get("train", loss.item() * cfg.gradient_accumulation_steps)
                val_loss = losses.get("val", 0.0)

                print(f"step {step:>6d} | train_loss {train_loss:.4f} | "
                      f"val_loss {val_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")

                # CSV log
                log_writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                     f"{lr:.6f}", f"{dt:.1f}"])
                log_file.flush()

                # Save checkpoint
                self.save_checkpoint(step, train_loss)

                # wandb
                if self.wandb_run:
                    import wandb
                    wandb.log({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": lr,
                        "step": step,
                    })

                t0 = time.time()

        log_file.close()

        # Save final model
        self.save_checkpoint(cfg.max_iters - 1, loss.item())
        # Also save tokenizer separately
        self.tokenizer.save(os.path.join(cfg.output_dir, "tokenizer.json"))
        print(f"\nTraining complete! Checkpoint saved to {cfg.checkpoint_path}")


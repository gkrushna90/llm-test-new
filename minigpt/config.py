"""
GPT Configuration - All hyperparameters in one place.
Supports small/medium/large presets for easy scaling.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class GPTConfig:
    """Configuration for GPT model, training, and data pipeline."""

    # --- Model Architecture ---
    n_embed: int = 128          # Embedding dimension
    n_heads: int = 4            # Number of attention heads
    n_layers: int = 4           # Number of transformer blocks
    block_size: int = 64        # Max sequence length (context window)
    dropout: float = 0.1        # Dropout rate (set to 0.0 for inference)
    vocab_size: int = 0         # Set by tokenizer (don't hardcode)
    bias: bool = False          # Use bias in Linear layers?

    # --- Training ---
    batch_size: int = 64
    max_iters: int = 10000
    eval_interval: int = 1000
    eval_iters: int = 200       # Number of batches for eval loss estimate
    learning_rate: float = 1e-3
    min_lr: float = 1e-4        # For cosine scheduler
    warmup_iters: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0      # Gradient clipping
    gradient_accumulation_steps: int = 1  # Simulate larger batch

    # --- Data ---
    data_dir: str = "data"
    train_split: float = 0.9

    # --- Paths ---
    output_dir: str = "outputs"
    checkpoint_path: str = "outputs/checkpoint.pth"
    log_path: str = "outputs/training_log.csv"

    # --- System ---
    device: str = ""            # Auto-detect if empty
    dtype: str = "float32"      # float32, float16, bfloat16
    compile_model: bool = False # torch.compile (PyTorch 2.0+)
    num_workers: int = 0        # DataLoader workers

    # --- Logging ---
    use_wandb: bool = False
    wandb_project: str = "minigpt"
    wandb_run_name: str = ""

    def __post_init__(self):
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def small(cls) -> "GPTConfig":
        """~2.5M params — good for CPU, small datasets."""
        return cls(
            n_embed=128,
            n_heads=4,
            n_layers=4,
            block_size=64,
            batch_size=64,
            max_iters=10000,
            learning_rate=1e-3,
        )

    @classmethod
    def medium(cls) -> "GPTConfig":
        """~25M params — needs GPU, medium datasets."""
        return cls(
            n_embed=384,
            n_heads=6,
            n_layers=6,
            block_size=256,
            batch_size=32,
            max_iters=20000,
            learning_rate=3e-4,
            dropout=0.2,
        )

    @classmethod
    def large(cls) -> "GPTConfig":
        """~125M params — needs good GPU, large datasets."""
        return cls(
            n_embed=768,
            n_heads=12,
            n_layers=12,
            block_size=512,
            batch_size=16,
            max_iters=50000,
            learning_rate=3e-4,
            dropout=0.2,
            gradient_accumulation_steps=4,
        )

    def param_count_estimate(self) -> int:
        """Rough estimate of model parameters."""
        # Embedding + Attention + FFN + output head
        embed = self.vocab_size * self.n_embed
        attn = self.n_layers * (4 * self.n_embed * self.n_embed)  # Q,K,V,proj
        ffn = self.n_layers * (2 * self.n_embed * 4 * self.n_embed)
        head = self.n_embed * self.vocab_size
        return embed + attn + ffn + head

    def summary(self) -> str:
        est = self.param_count_estimate()
        if est > 1e9:
            param_str = f"{est/1e9:.1f}B"
        elif est > 1e6:
            param_str = f"{est/1e6:.1f}M"
        else:
            param_str = f"{est/1e3:.1f}K"
        return (
            f"GPTConfig: {self.n_layers}L / {self.n_heads}H / {self.n_embed}D "
            f"| block_size={self.block_size} | ~{param_str} params "
            f"| device={self.device}"
        )


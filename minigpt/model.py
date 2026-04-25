"""
GPT Model — All components parameterized by GPTConfig.
Single source of truth for model architecture.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from minigpt.config import GPTConfig


class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.query = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.value = nn.Linear(config.n_embed, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        # Register causal mask as buffer (not a parameter, moves with .to(device))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embed // config.n_heads
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Feed-forward network with expansion factor of 4."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias),
            nn.GELU(),  # GELU is standard in modern transformers
            nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: attention + feed-forward with pre-norm."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Usage:
        config = GPTConfig.small()
        config.vocab_size = tokenizer.vocab_size
        model = GPT(config)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)

        # Weight tying (embed and output head share weights)
        self.head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T)
        )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


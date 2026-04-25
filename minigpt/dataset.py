"""
Dataset — Handles text data loading, tokenization, and batching.
Supports in-memory (small files) and streaming (large files).
"""

import os
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from minigpt.config import GPTConfig
from minigpt.tokenizer import Tokenizer


def _cache_path(data_dir: str, tokenizer_backend: str) -> str:
    """Generate a cache file path based on data files and tokenizer."""
    return os.path.join(data_dir, f".token_cache_{tokenizer_backend}.pt")


def _data_fingerprint(txt_files: list) -> str:
    """Hash file names + sizes + modification times to detect changes."""
    h = hashlib.md5()
    for fp in sorted(txt_files):
        stat = os.stat(fp)
        h.update(f"{fp}:{stat.st_size}:{stat.st_mtime}".encode())
    return h.hexdigest()


class TextDataset(Dataset):
    """
    In-memory text dataset. Good for files up to ~500MB.
    Tokenizes the entire file and serves random chunks.
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large files (GBs). Reads chunks without
    loading everything into memory.
    """

    def __init__(self, file_path: str, tokenizer: Tokenizer,
                 block_size: int, chunk_size: int = 1024 * 1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chunk_size = chunk_size  # Read 1MB at a time

    def __iter__(self):
        buffer = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                tokens = self.tokenizer.encode(chunk)
                buffer.extend(tokens)

                # Yield sequences from buffer
                while len(buffer) >= self.block_size + 1:
                    x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                    y = torch.tensor(buffer[1:self.block_size + 1], dtype=torch.long)
                    yield x, y
                    buffer = buffer[self.block_size:]


def _tokenize_and_cache(txt_files, tokenizer, cache_file, fingerprint):
    """Tokenize all text files and save to cache."""
    print(f"Tokenizing {len(txt_files)} file(s)...")
    all_text = ""
    for fp in txt_files:
        with open(fp, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
        print(f"  + {os.path.basename(fp)} ({os.path.getsize(fp):,} bytes)")

    print(f"  Total characters: {len(all_text):,}")

    tokens = tokenizer.encode(all_text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  Total tokens: {len(data):,}")

    # Save cache
    torch.save({"fingerprint": fingerprint, "tokens": data}, cache_file)
    print(f"  Cached tokens to {cache_file}")

    return data


def load_data(config: GPTConfig, tokenizer: Tokenizer, streaming: bool = False):
    """
    Load training and validation data.

    Args:
        config: GPTConfig with data_dir, train_split, block_size, batch_size
        tokenizer: Tokenizer instance
        streaming: If True, use StreamingTextDataset for large files

    Returns:
        train_loader, val_loader (DataLoaders)
    """
    # Find all .txt files in data directory
    data_dir = config.data_dir
    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ])

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    if streaming:
        # For streaming, we use the first file (extend for multi-file later)
        train_dataset = StreamingTextDataset(
            txt_files[0], tokenizer, config.block_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        return train_loader, None  # No val loader in streaming mode

    # In-memory mode: read all files (with caching)
    cache_file = _cache_path(data_dir, tokenizer.backend)
    fingerprint = _data_fingerprint(txt_files)

    # Check if cached tokens exist and data hasn't changed
    if os.path.exists(cache_file):
        cache = torch.load(cache_file)
        if cache.get("fingerprint") == fingerprint:
            data = cache["tokens"]
            print(f"Loaded cached tokens from {cache_file}")
            print(f"  {len(txt_files)} file(s) | {len(data):,} tokens (unchanged, skipped tokenization)")
        else:
            print(f"Data changed — re-tokenizing...")
            data = _tokenize_and_cache(txt_files, tokenizer, cache_file, fingerprint)
    else:
        data = _tokenize_and_cache(txt_files, tokenizer, cache_file, fingerprint)

    # Split
    n = int(config.train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = TextDataset(train_data, config.block_size)
    val_dataset = TextDataset(val_data, config.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    print(f"  Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    return train_loader, val_loader


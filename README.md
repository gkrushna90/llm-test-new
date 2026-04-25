# MiniGPT - Build a GPT from Scratch

A scalable GPT implementation for learning and experimentation.

## Project Structure

```
model-v1.0/
├── minigpt/                    # Core package
│   ├── __init__.py             # Package init
│   ├── config.py               # All hyperparameters + size presets
│   ├── model.py                # GPT model (Head, Attention, Block, GPT)
│   ├── tokenizer.py            # Char-level, tiktoken, or SentencePiece
│   ├── dataset.py              # In-memory + streaming data loading
│   ├── trainer.py              # Training loop, checkpoints, logging
│   ├── generate.py             # Text generation (temp, top-k, top-p)
│   └── cli.py                  # CLI commands: train, generate, chat
├── data/                       # Put your .txt training files here
│   └── input.txt
├── outputs/                    # Checkpoints, logs, tokenizer (auto-created)
├── requirements.txt
├── README.md
├── train_gpt.py                # [Legacy] Old monolithic training script
└── main.py                     # [Legacy] Old inference script
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add training data
Put `.txt` files in the `data/` folder.

### 3. Train
```bash
# Small model (CPU friendly, ~2.5M params)
python -m minigpt.cli train --size small --data data/

# Medium model (needs GPU, ~25M params)
python -m minigpt.cli train --size medium --data data/

# Large model (needs good GPU, ~125M params)
python -m minigpt.cli train --size large --data data/

# Custom options
python -m minigpt.cli train --size small --iters 20000 --tokenizer tiktoken --wandb
```

### 4. Chat
```bash
python -m minigpt.cli chat
```

### 5. Generate
```bash
python -m minigpt.cli generate --prompt "Q: What is the capital of France?"
python -m minigpt.cli generate --prompt "Once upon a time" --temperature 1.0 --top-k 50
```

## Model Sizes

| Size | Params | Layers | Heads | Embed | Context | Hardware |
|------|--------|--------|-------|-------|---------|----------|
| small | ~2.5M | 4 | 4 | 128 | 64 | CPU |
| medium | ~25M | 6 | 6 | 384 | 256 | GPU |
| large | ~125M | 12 | 12 | 768 | 512 | GPU |

## Tokenizer Options

| Backend | Vocab Size | Training | Best For |
|---------|-----------|----------|----------|
| `char` | ~70 | None | Learning, tiny datasets |
| `tiktoken` | 50,257 | None | Medium/large datasets |
| `sentencepiece` | Custom | Trains on your data | Domain-specific |

```bash
# Use GPT-2 BPE tokenizer (pip install tiktoken)
python -m minigpt.cli train --tokenizer tiktoken

# Train custom BPE (pip install sentencepiece)  
python -m minigpt.cli train --tokenizer sentencepiece
```

## Features

- **Resumable training** — checkpoints save every N steps
- **Cosine LR schedule** with warmup
- **Gradient clipping** and accumulation
- **Mixed precision** (float16/bfloat16 on GPU)
- **Streaming dataset** for large files (--streaming)
- **wandb integration** (--wandb)
- **Top-k / Top-p sampling** for generation
- **Weight tying** (embedding and output head share weights)
- **Pre-LayerNorm** transformer blocks


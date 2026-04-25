"""
CLI — Command-line interface for training, generation, and chat.

Usage:
    python -m minigpt.cli train --size small --data data/
    python -m minigpt.cli chat
    python -m minigpt.cli generate --prompt "Q: What is 5+5?"
"""

import argparse
import os
import sys
import torch

from minigpt.config import GPTConfig
from minigpt.model import GPT
from minigpt.tokenizer import Tokenizer
from minigpt.dataset import load_data
from minigpt.trainer import Trainer
from minigpt.generate import generate, generate_qa


def cmd_train(args):
    """Train the model."""
    # Config
    if args.size == "small":
        config = GPTConfig.small()
    elif args.size == "medium":
        config = GPTConfig.medium()
    elif args.size == "large":
        config = GPTConfig.large()
    else:
        config = GPTConfig.small()

    # Override from CLI args
    if args.data:
        config.data_dir = args.data
    if args.iters:
        config.max_iters = args.iters
    if args.lr:
        config.learning_rate = args.lr
    if args.wandb:
        config.use_wandb = True
    if args.output:
        config.output_dir = args.output
        config.checkpoint_path = os.path.join(args.output, "checkpoint.pth")
        config.log_path = os.path.join(args.output, "training_log.csv")

    # Tokenizer
    if args.tokenizer == "tiktoken":
        tokenizer = Tokenizer.from_pretrained("tiktoken")
    elif args.tokenizer == "sentencepiece":
        sp_model = os.path.join(config.output_dir, "sp_tokenizer.model")
        if os.path.exists(sp_model):
            tokenizer = Tokenizer.train_sentencepiece.__func__  # load existing
        else:
            txt_files = [os.path.join(config.data_dir, f) for f in os.listdir(config.data_dir) if f.endswith('.txt')]
            tokenizer = Tokenizer.train_sentencepiece(txt_files[0], model_prefix=os.path.join(config.output_dir, "sp_tokenizer"))
    else:
        # Char-level: read all text to build vocab
        all_text = ""
        for f in sorted(os.listdir(config.data_dir)):
            if f.endswith(".txt"):
                with open(os.path.join(config.data_dir, f), "r", encoding="utf-8") as fh:
                    all_text += fh.read() + "\n"
        tokenizer = Tokenizer.from_text(all_text, backend="char")

    config.vocab_size = tokenizer.vocab_size
    print(f"\n{config.summary()}")

    # Data
    train_loader, val_loader = load_data(config, tokenizer, streaming=args.streaming)

    # Model
    model = GPT(config).to(config.device)
    print(f"Model parameters: {model.count_parameters():,}\n")

    # Compile (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Train
    trainer = Trainer(config, model, tokenizer, train_loader, val_loader)
    trainer.train()


def cmd_generate(args):
    """Generate text from a prompt."""
    config, model, tokenizer = _load_model(args.checkpoint)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=config.device,
    )

    if args.prompt:
        print(args.prompt + output)
    else:
        print(output)


def cmd_chat(args):
    """Interactive Q&A chat mode."""
    config, model, tokenizer = _load_model(args.checkpoint)

    print("\n" + "=" * 50)
    print("  MiniGPT Chat")
    print("=" * 50)
    print("  Type a question and press Enter.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = generate_qa(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=config.device,
            temperature=args.temperature,
        )
        print(f"AI: {answer}\n")


def _load_model(checkpoint_path: str = "outputs/checkpoint.pth"):
    """Load model and tokenizer from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Train the model first: python -m minigpt.cli train")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Rebuild config
    saved_config = checkpoint.get("config", {})
    config = GPTConfig(**{k: v for k, v in saved_config.items()
                          if k in GPTConfig.__dataclass_fields__})
    config.dropout = 0.0  # No dropout during inference

    # Rebuild tokenizer
    backend = checkpoint.get("tokenizer_backend", "char")
    if backend == "char":
        tokenizer = Tokenizer.from_checkpoint(checkpoint)
    elif backend == "tiktoken":
        tokenizer = Tokenizer.from_pretrained("tiktoken")
    else:
        tokenizer = Tokenizer.from_checkpoint(checkpoint)

    # Detect if old checkpoint used bias (backward compatibility)
    has_bias = any("proj.bias" in k for k in checkpoint["model_state"].keys())
    if has_bias:
        config.bias = True

    config.vocab_size = tokenizer.vocab_size

    # Build model
    model = GPT(config).to(config.device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    step = checkpoint.get("step", checkpoint.get("iter", "?"))
    loss = checkpoint.get("loss", 0)
    print(f"Model loaded (step {step}, loss {loss:.4f})")
    print(config.summary())

    return config, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniGPT - Train and use a GPT model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    train_parser.add_argument("--data", type=str, default="data", help="Data directory")
    train_parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    train_parser.add_argument("--iters", type=int, default=None, help="Max training iterations")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--tokenizer", choices=["char", "tiktoken", "sentencepiece"], default="char")
    train_parser.add_argument("--streaming", action="store_true", help="Stream data from disk")
    train_parser.add_argument("--wandb", action="store_true", help="Log to wandb")

    # --- Generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--prompt", type=str, default="", help="Starting text")
    gen_parser.add_argument("--tokens", type=int, default=200, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7)
    gen_parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling")
    gen_parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint.pth")

    # --- Chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A chat")
    chat_parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint.pth")
    chat_parser.add_argument("--temperature", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


"""
Tokenizer — Supports character-level, tiktoken (GPT-2 BPE), or custom SentencePiece.
Switchable backend so you can start simple and scale up.
"""

import os
import json
from typing import List


class Tokenizer:
    """
    Unified tokenizer interface.

    Backends:
        - "char"        : character-level (simple, small vocab ~70)
        - "tiktoken"    : GPT-2 BPE tokenizer (50257 tokens, no training needed)
        - "sentencepiece": Custom BPE trained on your data

    Usage:
        tok = Tokenizer.from_text(text, backend="char")
        tok = Tokenizer.from_pretrained("tiktoken")
        encoded = tok.encode("Hello world")
        decoded = tok.decode(encoded)
        tok.save("outputs/tokenizer.json")
        tok = Tokenizer.load("outputs/tokenizer.json")
    """

    def __init__(self, backend: str = "char", stoi: dict = None, itos: dict = None):
        self.backend = backend
        self.stoi = stoi or {}
        self.itos = itos or {}
        self._tiktoken_enc = None
        self._sp_model = None

    @property
    def vocab_size(self) -> int:
        if self.backend == "tiktoken":
            return self._tiktoken_enc.n_vocab
        elif self.backend == "sentencepiece":
            return self._sp_model.get_piece_size()
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        if self.backend == "tiktoken":
            return self._tiktoken_enc.encode(text)
        elif self.backend == "sentencepiece":
            return self._sp_model.encode(text)
        # char-level
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        if self.backend == "tiktoken":
            return self._tiktoken_enc.decode(tokens)
        elif self.backend == "sentencepiece":
            return self._sp_model.decode(tokens)
        # char-level
        return ''.join([self.itos.get(i, '') for i in tokens])

    # ---- Constructors ----

    @classmethod
    def from_text(cls, text: str, backend: str = "char") -> "Tokenizer":
        """Build a character-level tokenizer from raw text."""
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(backend="char", stoi=stoi, itos=itos)

    @classmethod
    def from_pretrained(cls, backend: str = "tiktoken") -> "Tokenizer":
        """Load a pretrained tokenizer (tiktoken GPT-2 or sentencepiece)."""
        tok = cls(backend=backend)

        if backend == "tiktoken":
            try:
                import tiktoken
                tok._tiktoken_enc = tiktoken.get_encoding("gpt2")
            except ImportError:
                raise ImportError("Install tiktoken: pip install tiktoken")

        elif backend == "sentencepiece":
            raise NotImplementedError(
                "Use Tokenizer.train_sentencepiece() to create a custom tokenizer"
            )

        return tok

    @classmethod
    def train_sentencepiece(cls, data_path: str, vocab_size: int = 4000,
                            model_prefix: str = "outputs/sp_tokenizer") -> "Tokenizer":
        """Train a SentencePiece BPE tokenizer on your data."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Install sentencepiece: pip install sentencepiece")

        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=3,
        )

        tok = cls(backend="sentencepiece")
        tok._sp_model = spm.SentencePieceProcessor()
        tok._sp_model.load(f"{model_prefix}.model")
        return tok

    # ---- Save / Load ----

    def save(self, path: str):
        """Save tokenizer to JSON (char-level) or note the backend."""
        data = {
            "backend": self.backend,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from JSON or checkpoint."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        backend = data.get("backend", "char")
        tok = cls(backend=backend)

        if backend == "char":
            tok.stoi = data["stoi"]
            tok.itos = {int(k): v for k, v in data["itos"].items()}
        elif backend == "tiktoken":
            tok = cls.from_pretrained("tiktoken")
        elif backend == "sentencepiece":
            # Expects sp_tokenizer.model in same directory
            sp_dir = os.path.dirname(path)
            import sentencepiece as spm
            tok._sp_model = spm.SentencePieceProcessor()
            tok._sp_model.load(os.path.join(sp_dir, "sp_tokenizer.model"))

        return tok

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "Tokenizer":
        """Load tokenizer from a training checkpoint dict (backward compatible)."""
        tok = cls(backend="char")
        tok.stoi = checkpoint.get("stoi", {})
        itos_raw = checkpoint.get("itos", {})
        tok.itos = {int(k): v for k, v in itos_raw.items()}
        return tok


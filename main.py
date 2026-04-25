# =========================
# GPT Inference Script
# Load the trained model and generate text
# =========================

import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================
# Hyperparameters (must match train_gpt.py)
# =========================
block_size = 64
n_embed = 128
n_heads = 4
n_layers = 4
dropout = 0.0  # no dropout during inference

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer and vocab_size will be loaded from checkpoint before model is built
stoi = {}
itos = {}
vocab_size = 0  # will be updated after loading checkpoint


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(l):
    return ''.join([itos[i] for i in l])


# =========================
# Load Checkpoint & Tokenizer FIRST (before building model)
# =========================
import os

if os.path.exists("gpt_checkpoint.pth"):
    checkpoint = torch.load("gpt_checkpoint.pth", map_location=device)
    stoi.update(checkpoint["stoi"])
    itos.update({int(k): v for k, v in checkpoint["itos"].items()})
    vocab_size = len(stoi)
    print(f"Tokenizer loaded. Vocab size: {vocab_size} characters")
elif os.path.exists("gpt_model.pth") and os.path.getsize("gpt_model.pth") > 0:
    checkpoint = None
    print("WARNING: Using gpt_model.pth — no tokenizer embedded. Falling back to input.txt")
    with open("input.txt", "r", encoding="utf-8") as f:
        _text = f.read()
    _chars = sorted(list(set(_text)))
    vocab_size = len(_chars)
    stoi.update({ch: i for i, ch in enumerate(_chars)})
    itos.update({i: ch for i, ch in enumerate(_chars)})
else:
    print("ERROR: No valid model file found. Train the model first using train_gpt.py")
    exit()

# =========================
# Build Model (now vocab_size is correct)
# =========================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=device))
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


# =========================
# Load Model Weights
# =========================
model = GPT().to(device)

if os.path.exists("gpt_checkpoint.pth"):
    model.load_state_dict(checkpoint["model_state"])
    print(f"Model loaded from checkpoint (step {checkpoint['iter']}, loss {checkpoint['loss']:.4f})")
else:
    model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
    print("Model loaded from gpt_model.pth")

model.eval()


# =========================
# Text Generation Function
# =========================
def generate(prompt="", max_new_tokens=300, temperature=1.0):
    """
    Generate text from the model. Stops after the first complete answer.
    """
    if prompt:
        encoded = encode(prompt)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_text = ""
    found_answer = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            # Decode just the new character
            char = decode([next_token.item()])
            generated_text += char

            # Stop after first complete answer (A: ... followed by double newline or next Q:)
            if found_answer and ("Q:" in generated_text.split("\n")[-1]):
                # Remove the trailing "Q:..." part
                lines = generated_text.strip().split("\n")
                # Keep only lines up to the answer
                result_lines = []
                for line in lines:
                    if line.startswith("Q:") and result_lines:
                        break
                    result_lines.append(line)
                return "\n".join(result_lines)

            if "A:" in generated_text:
                found_answer = True

    return generated_text.strip()


# =========================
# Interactive Mode
# =========================
print("\n" + "="*50)
print("  GPT Q&A Model - Ask me anything!")
print("="*50)
print("Tips:")
print("  Type a question     -> get an answer")
print("  'quit' or 'exit'    -> stop")
print("  Questions work best starting with 'Q: ' or 'What is'")
print()

while True:
    prompt = input("You: ").strip()

    if prompt.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if not prompt:
        continue

    # Auto-add "Q: " prefix if not present
    if not prompt.startswith("Q:"):
        prompt = "Q: " + prompt

    # Add newline + "A:" to guide the model to answer
    if not prompt.endswith("\nA:"):
        full_prompt = prompt + "\nA:"
    else:
        full_prompt = prompt

    output = generate(prompt=full_prompt, max_new_tokens=200, temperature=0.5)
    # Clean up: show just the answer part
    answer = output.strip()
    if answer.startswith(":"):
        answer = answer[1:].strip()
    print(f"AI: {answer}")
    print()

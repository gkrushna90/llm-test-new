# =========================
# 1. Imports
# =========================
import os
import csv
import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================
# 2. Hyperparameters
# =========================
batch_size = 64
block_size = 64
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-3

n_embed = 128
n_heads = 4
n_layers = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "gpt_checkpoint.pth"
LOG_PATH = "training_log.csv"

# =========================
# 3. Load dataset
# =========================
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# 4. Tokenizer (char-level)
# =========================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
print("Dataset length:", len(data))
print("Block size:", block_size)
print("Sample text:", text[:100])

# =========================
# 5. Train/val split
# =========================
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


# =========================
# 6. Model components
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
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(n_heads)]
        )
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
        loss = F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T)
        )

        return logits, loss


# =========================
# 7. Initialize model
# =========================
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_iter = 0

# Resume from checkpoint if available
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_iter = checkpoint["iter"] + 1
    print(f"Resumed from checkpoint at step {checkpoint['iter']}")
else:
    print("No checkpoint found, starting fresh.")

# Prepare log file
log_file_exists = os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if not log_file_exists:
    log_writer.writerow(["step", "train_loss"])

# =========================
# 8. Training loop
# =========================
for iter in range(start_iter, max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        loss_val = loss.item()
        print(f"step {iter}: loss {loss_val:.4f}")

        # Log to file
        log_writer.writerow([iter, f"{loss_val:.4f}"])
        log_file.flush()

        # Save checkpoint (includes tokenizer so main.py doesn't need input.txt)
        torch.save({
            "iter": iter,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss_val,
            "stoi": stoi,
            "itos": itos,
            "vocab_size": vocab_size,
        }, CHECKPOINT_PATH)
        print(f"  -> Checkpoint saved at step {iter}")

log_file.close()

# Save final model
torch.save(model.state_dict(), "gpt_model.pth")
print("Model saved!")
print(f"Training log saved to {LOG_PATH}")

# =========================
# 9. Text generation
# =========================
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx


context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = generate(model, context, 300)

print(decode(generated[0].tolist()))

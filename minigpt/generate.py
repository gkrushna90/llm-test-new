"""
Text Generation — Supports temperature, top-k, and top-p (nucleus) sampling.
"""

import torch
from torch.nn import functional as F

from minigpt.model import GPT
from minigpt.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.0,
    stop_at: str = None,
    device: str = "cpu",
) -> str:
    """
    Generate text from the model.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        prompt: Starting text (empty = generate from scratch)
        max_new_tokens: Max characters/tokens to generate
        temperature: Sampling temperature (lower = more focused)
        top_k: If > 0, only sample from top-k tokens
        top_p: If > 0, use nucleus sampling (sample from smallest set with cumprob > p)
        stop_at: Stop generation when this string appears in output
        device: Device to run on

    Returns:
        Generated text string (excluding the prompt)
    """
    model.eval()
    block_size = model.config.block_size

    if prompt:
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_text = ""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # Last token logits

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        # Decode new character
        char = tokenizer.decode([next_token.item()])
        generated_text += char

        # Stop condition
        if stop_at and stop_at in generated_text:
            # Trim after stop marker
            generated_text = generated_text[:generated_text.index(stop_at)]
            break

    return generated_text


def generate_qa(model: GPT, tokenizer: Tokenizer, question: str,
                device: str = "cpu", temperature: float = 0.5) -> str:
    """
    Generate a single Q&A answer. Stops after the first answer.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        question: The question to ask
        device: Device to run on
        temperature: Sampling temperature

    Returns:
        The answer string
    """
    # Format prompt
    if not question.startswith("Q:"):
        question = "Q: " + question
    prompt = question + "\nA:"

    # Generate and stop at next "Q:"
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        temperature=temperature,
        stop_at="\nQ:",
        device=device,
    )

    answer = output.strip()
    if answer.startswith(":"):
        answer = answer[1:].strip()
    return answer


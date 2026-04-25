"""
Download math & computation training data from public sources.
Run: python download_math_data.py
"""

import os
import random

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def generate_arithmetic(filename="math_arithmetic.txt", n=50000):
    """Generate arithmetic Q&A pairs: addition, subtraction, multiplication, division."""
    print(f"Generating {n} arithmetic problems...")
    lines = []

    for _ in range(n):
        op = random.choice(["+", "-", "*", "/"])
        if op == "+":
            a, b = random.randint(0, 999), random.randint(0, 999)
            ans = a + b
            lines.append(f"Q: What is {a}+{b}?\nA: {a}+{b}={ans}.\n")
        elif op == "-":
            a = random.randint(0, 999)
            b = random.randint(0, a)  # keep positive
            ans = a - b
            lines.append(f"Q: What is {a}-{b}?\nA: {a}-{b}={ans}.\n")
        elif op == "*":
            a, b = random.randint(0, 99), random.randint(0, 99)
            ans = a * b
            lines.append(f"Q: What is {a}*{b}?\nA: {a}*{b}={ans}.\n")
        elif op == "/":
            b = random.randint(1, 50)
            ans = random.randint(1, 50)
            a = b * ans  # ensure clean division
            lines.append(f"Q: What is {a}/{b}?\nA: {a}/{b}={ans}.\n")

    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {len(lines)} problems to {path} ({os.path.getsize(path):,} bytes)")


def generate_number_facts(filename="math_facts.txt"):
    """Generate number facts: squares, cubes, primes, fibonacci, etc."""
    print("Generating number facts...")
    lines = []

    # Squares
    for n in range(1, 101):
        lines.append(f"Q: What is {n} squared?\nA: {n} squared is {n*n}.\n")
        lines.append(f"Q: What is {n}*{n}?\nA: {n}*{n}={n*n}.\n")

    # Cubes
    for n in range(1, 51):
        lines.append(f"Q: What is {n} cubed?\nA: {n} cubed is {n**3}.\n")

    # Square roots (perfect squares)
    for n in range(1, 101):
        lines.append(f"Q: What is the square root of {n*n}?\nA: The square root of {n*n} is {n}.\n")

    # Powers of 2
    for n in range(1, 21):
        lines.append(f"Q: What is 2 to the power of {n}?\nA: 2 to the power of {n} is {2**n}.\n")

    # Powers of 10
    for n in range(1, 11):
        lines.append(f"Q: What is 10 to the power of {n}?\nA: 10 to the power of {n} is {10**n}.\n")

    # Factorials
    fact = 1
    for n in range(1, 13):
        fact *= n
        lines.append(f"Q: What is {n} factorial?\nA: {n} factorial is {fact}.\n")
        lines.append(f"Q: What is {n}!?\nA: {n}!={fact}.\n")

    # Fibonacci
    fib = [0, 1]
    for i in range(2, 21):
        fib.append(fib[-1] + fib[-2])
    for i, v in enumerate(fib):
        lines.append(f"Q: What is the {i+1}th Fibonacci number?\nA: The {i+1}th Fibonacci number is {v}.\n")

    # Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    primes = [n for n in range(2, 200) if is_prime(n)]
    for p in primes:
        lines.append(f"Q: Is {p} a prime number?\nA: Yes, {p} is a prime number.\n")

    # Non-primes
    for n in [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 33, 35, 36]:
        lines.append(f"Q: Is {n} a prime number?\nA: No, {n} is not a prime number.\n")

    # Even/odd
    for n in range(1, 101):
        if n % 2 == 0:
            lines.append(f"Q: Is {n} even or odd?\nA: {n} is even.\n")
        else:
            lines.append(f"Q: Is {n} even or odd?\nA: {n} is odd.\n")

    # Percentages
    for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        for base in [50, 100, 200, 500, 1000]:
            ans = int(pct * base / 100)
            lines.append(f"Q: What is {pct}% of {base}?\nA: {pct}% of {base} is {ans}.\n")

    # Repeat 5x for reinforcement
    lines = lines * 5
    random.shuffle(lines)

    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {len(lines)} facts to {path} ({os.path.getsize(path):,} bytes)")


def generate_word_problems(filename="math_word_problems.txt"):
    """Generate simple word problems."""
    print("Generating word problems...")
    lines = []

    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Harry", "Ivy", "Jack"]
    items = ["apples", "books", "coins", "pencils", "oranges", "marbles", "candies", "stickers", "flowers", "cookies"]

    for _ in range(5000):
        name = random.choice(names)
        item = random.choice(items)

        # Addition word problem
        a, b = random.randint(1, 50), random.randint(1, 50)
        lines.append(f"Q: {name} has {a} {item} and gets {b} more. How many {item} does {name} have?\nA: {name} has {a}+{b}={a+b} {item}.\n")

        # Subtraction word problem
        a = random.randint(10, 100)
        b = random.randint(1, a)
        lines.append(f"Q: {name} has {a} {item} and gives away {b}. How many {item} does {name} have left?\nA: {name} has {a}-{b}={a-b} {item} left.\n")

        # Multiplication word problem
        a, b = random.randint(1, 20), random.randint(1, 20)
        lines.append(f"Q: {name} buys {a} boxes with {b} {item} each. How many {item} in total?\nA: {name} has {a}*{b}={a*b} {item} in total.\n")

        # Division word problem
        b = random.randint(2, 10)
        a = b * random.randint(1, 20)
        lines.append(f"Q: {name} shares {a} {item} equally among {b} friends. How many does each get?\nA: Each friend gets {a}/{b}={a//b} {item}.\n")

    random.shuffle(lines)

    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {len(lines)} word problems to {path} ({os.path.getsize(path):,} bytes)")


def generate_conversions(filename="math_conversions.txt"):
    """Generate unit conversion Q&A pairs."""
    print("Generating unit conversions...")
    lines = []

    # Temperature
    for c in range(0, 101, 5):
        f = round(c * 9/5 + 32, 1)
        lines.append(f"Q: What is {c} Celsius in Fahrenheit?\nA: {c} Celsius is {f} Fahrenheit.\n")

    for f in range(32, 213, 10):
        c = round((f - 32) * 5/9, 1)
        lines.append(f"Q: What is {f} Fahrenheit in Celsius?\nA: {f} Fahrenheit is {c} Celsius.\n")

    # Length
    for km in range(1, 51):
        mi = round(km * 0.621371, 2)
        lines.append(f"Q: How many miles is {km} km?\nA: {km} km is {mi} miles.\n")

    for m in [1, 2, 3, 5, 10, 20, 50, 100]:
        ft = round(m * 3.28084, 2)
        lines.append(f"Q: How many feet is {m} meters?\nA: {m} meters is {ft} feet.\n")

    # Weight
    for kg in range(1, 51):
        lb = round(kg * 2.20462, 2)
        lines.append(f"Q: How many pounds is {kg} kg?\nA: {kg} kg is {lb} pounds.\n")

    # Time
    for h in range(1, 25):
        lines.append(f"Q: How many minutes is {h} hours?\nA: {h} hours is {h*60} minutes.\n")
        lines.append(f"Q: How many seconds is {h} hours?\nA: {h} hours is {h*3600} seconds.\n")

    lines = lines * 5
    random.shuffle(lines)

    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {len(lines)} conversions to {path} ({os.path.getsize(path):,} bytes)")


def download_gsm8k():
    """Download GSM8K — 8,500 real grade school math problems."""
    print("Downloading GSM8K (grade school math)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        path = os.path.join(DATA_DIR, "gsm8k.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in ds:
                q = item["question"].strip()
                a = item["answer"].strip()
                f.write(f"Q: {q}\nA: {a}\n\n")
        print(f"  Saved {len(ds)} problems to {path} ({os.path.getsize(path):,} bytes)")
    except Exception as e:
        print(f"  Could not download GSM8K: {e}")
        print("  (Run: pip install datasets)")


if __name__ == "__main__":
    print("=" * 50)
    print("  Downloading Math & Computation Data")
    print("=" * 50)
    print()

    random.seed(42)

    generate_arithmetic(n=50000)
    generate_number_facts()
    generate_word_problems()
    generate_conversions()
    download_gsm8k()

    print("\n" + "=" * 50)
    total = sum(os.path.getsize(os.path.join(DATA_DIR, f))
                for f in os.listdir(DATA_DIR) if f.endswith(".txt"))
    count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
    print(f"Done! {count} files, {total:,} bytes ({total/1024/1024:.1f} MB) in data/")
    print("Run: python -m minigpt.cli train --size small --data data/")


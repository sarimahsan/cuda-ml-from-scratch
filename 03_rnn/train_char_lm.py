"""
=============================================================================
Character-Level Language Model Training using Pure CUDA RNN
=============================================================================
Trains a character language model on Shakespeare text (or synthetic text)
using CUDARNNLanguageModel and custom CUDA primitives.
=============================================================================
"""

import argparse
import math
import sys
import time
import torch
import torch.nn as nn
from rnn import CUDARNNLanguageModel, _ext


SAMPLE_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!
""" * 20


def build_dataset(text: str, seq_len: int = 32, batch_size: int = 32):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    num_batches = (len(data) - 1) // (seq_len * batch_size)
    trimmed_len = num_batches * seq_len * batch_size
    inputs = data[:trimmed_len].view(batch_size, -1)
    targets = data[1:trimmed_len + 1].view(batch_size, -1)

    return inputs, targets, vocab_size, char_to_idx, idx_to_char, num_batches


def main():
    parser = argparse.ArgumentParser(description="Train CUDA Character RNN Language Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to train this model.")
        sys.exit(1)

    inputs, targets, vocab_size, char_to_idx, idx_to_char, num_batches = build_dataset(
        SAMPLE_TEXT, seq_len=args.seq_len, batch_size=args.batch_size
    )

    print("=" * 75)
    print(" 📜 TRAINING CHARACTER RNN LANGUAGE MODEL (PURE CUDA) 📜")
    print(f" • Device        : {torch.cuda.get_device_name(0)}")
    print(f" • Vocabulary    : {vocab_size} unique characters")
    print(f" • Architecture  : Embed({args.embed_dim}) -> CUDARNN({args.hidden_size}) -> Linear({vocab_size})")
    print(f" • Sequence Len  : {args.seq_len} timesteps | Batch Size: {args.batch_size}")
    print("=" * 75)

    model = CUDARNNLanguageModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        nonlinearity="tanh"
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    inputs = inputs.to(args.device)
    targets = targets.to(args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.perf_counter()

        for b in range(num_batches):
            x_batch = inputs[:, b * args.seq_len : (b + 1) * args.seq_len].t().contiguous() # [T, N]
            y_batch = targets[:, b * args.seq_len : (b + 1) * args.seq_len].t().contiguous() # [T, N]

            optimizer.zero_grad()
            logits, _ = model(x_batch) # [T, N, V]
            loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.perf_counter() - start_t
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:.4f} | Time: {elapsed*1000:.1f} ms")

    print("\n" + "=" * 75)
    print(" 🎭 TEXT GENERATION SAMPLE (Temperature = 0.8)")
    print("=" * 75)
    seed = [char_to_idx[c] for c in "First Citizen:"]
    generated_indices = model.generate(seed, max_length=150, temperature=0.8)
    generated_text = "".join([idx_to_char[i] for i in generated_indices])
    print(generated_text)
    print("=" * 75)


if __name__ == "__main__":
    main()

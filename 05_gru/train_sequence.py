"""
=============================================================================
CUDA GRU Character-Level Language Model Training (Shakespeare)
=============================================================================
Trains a pure CUDA GRU language model end-to-end on text sequences:
  - Vocabulary tokenization & embedding
  - Sequence BPTT backpropagation
  - Fused Adam optimizer step
  - Text generation & temperature sampling
=============================================================================
"""

import argparse
import math
import os
import sys
import time
import torch
import torch.nn as nn

try:
    from gru import CUDAGRULanguageModel
except ImportError as e:
    print(f"[ERROR] Failed to import CUDA GRU Language Model: {e}")
    sys.exit(1)

SHAKESPEARE_SAMPLE = """
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

Second Citizen:
One word, good citizens.
"""


def load_dataset(data_path: str = None) -> str:
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return f.read()
    return SHAKESPEARE_SAMPLE * 150  # Repeat to build working corpus


def main():
    parser = argparse.ArgumentParser(description="Train CUDA GRU Character Language Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Sequence batch size")
    parser.add_argument("--seq_len", type=int, default=48, help="Sequence length T")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=128, help="GRU hidden state dimension")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is required for training.")
        sys.exit(1)

    print("=" * 75)
    print(" 📜 CUDA ML: GATED RECURRENT UNIT (GRU) LANGUAGE MODEL TRAINING 📜")
    print(f" • GPU Device  : {torch.cuda.get_device_name(0)}")
    print(f" • Epochs      : {args.epochs} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")
    print(f" • Embed Dim   : {args.embed_dim} | Hidden Size: {args.hidden_size} | LR: {args.lr}")
    print("=" * 75)

    text = load_dataset()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

    print(f"[INFO] Corpus Length: {len(text):,} characters | Vocabulary Size: {vocab_size}")

    data_indices = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    num_batches = (len(data_indices) - 1) // (args.batch_size * args.seq_len)

    model = CUDAGRULanguageModel(vocab_size, args.embed_dim, args.hidden_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Beginning CUDA GRU Language Model Training...\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for b in range(num_batches):
            start_idx = b * args.batch_size * args.seq_len
            x_chunk = torch.stack([
                data_indices[start_idx + i * args.seq_len : start_idx + (i + 1) * args.seq_len]
                for i in range(args.batch_size)
            ], dim=1).to(args.device) # [T, N]

            y_chunk = torch.stack([
                data_indices[start_idx + i * args.seq_len + 1 : start_idx + (i + 1) * args.seq_len + 1]
                for i in range(args.batch_size)
            ], dim=1).to(args.device) # [T, N]

            optimizer.zero_grad()
            logits, _ = model(x_chunk) # [T, N, V]

            loss = criterion(logits.view(-1, vocab_size), y_chunk.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_ms = (time.time() - t0) * 1000.0
        print(f"  Epoch {epoch:2d}/{args.epochs:2d} | Cross-Entropy Loss: {avg_loss:.4f} | Perplexity: {math.exp(avg_loss):.2f} | Time: {epoch_ms:.2f} ms")

    total_duration = time.time() - start_time
    print("-" * 75)
    print(f"[SUCCESS] Training complete in {total_duration:.2f}s ({total_duration / args.epochs * 1000.0:.2f} ms/epoch)")

    # Text generation test
    print("\n" + "=" * 75)
    print(" 🎭 SAMPLE GENERATION DEMO (Seed: 'First Citizen:')")
    print("=" * 75)
    seed = "First Citizen:"
    seed_idx = [char2idx.get(c, 0) for c in seed]
    generated_indices = model.generate(seed_idx, max_length=150, temperature=0.7)
    generated_text = "".join([idx2char[i] for i in generated_indices])
    print(generated_text)
    print("=" * 75)


if __name__ == "__main__":
    main()

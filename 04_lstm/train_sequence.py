"""
=============================================================================
Modular Pure CUDA LSTM: Real Character-Level Language Model Training
=============================================================================
Trains a character-level sequence model on Shakespeare text using our
pure CUDA LSTM kernels with Backpropagation Through Time (BPTT).
=============================================================================
"""

import math
import os
import sys
import time
import urllib.request
import torch

from lstm import CUDALSTMLanguageModel


SHAKESPEARE_FALLBACK = """
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

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            print("[INFO] Downloading Tiny Shakespeare dataset...")
            urllib.request.urlretrieve(url, data_path)
        except Exception as e:
            print(f"[WARNING] Could not download dataset ({e}). Using embedded Shakespeare corpus.")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(SHAKESPEARE_FALLBACK * 50)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    encoded = [char2idx[ch] for ch in text]
    return text, encoded, char2idx, idx2char


def get_batches(data, batch_size=32, seq_len=64, device="cuda"):
    num_batches = (len(data) - 1) // (batch_size * seq_len)
    if num_batches == 0:
        # Repeat data if too short
        data = data * ((batch_size * seq_len * 2) // len(data) + 1)
        num_batches = (len(data) - 1) // (batch_size * seq_len)

    total_len = num_batches * batch_size * seq_len
    x_data = torch.tensor(data[:total_len], dtype=torch.long, device=device)
    y_data = torch.tensor(data[1 : total_len + 1], dtype=torch.long, device=device)

    x_batches = x_data.view(batch_size, -1).t().contiguous()
    y_batches = y_data.view(batch_size, -1).t().contiguous()

    for i in range(0, x_batches.size(0) - seq_len + 1, seq_len):
        yield x_batches[i : i + seq_len], y_batches[i : i + seq_len]


def main():
    print("=" * 75)
    print("   📜 CUDA ML FROM SCRATCH: LSTM Sequence Language Model Training   ")
    print("=" * 75)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on this system. Please run on a GPU server.")
        return

    device = "cuda"
    text, encoded, char2idx, idx2char = load_dataset()
    vocab_size = len(char2idx)

    hidden_dim = 256
    seq_len = 64
    batch_size = 64
    learning_rate = 2e-3
    epochs = 5

    print(f"• Dataset Length   : {len(text):,} characters")
    print(f"• Vocabulary Size  : {vocab_size} unique characters")
    print(f"• Hidden Dimension : {hidden_dim}")
    print(f"• Sequence Length  : {seq_len}")
    print(f"• Batch Size       : {batch_size}")
    print(f"• Learning Rate    : {learning_rate}")
    print(f"• Target Device    : {torch.cuda.get_device_name(0)}")
    print("-" * 75)

    model = CUDALSTMLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        clip_norm=1.0,
        mode="fused",
        device=device,
    )

    seed_text = "KING:\nTo be or not"
    seed_indices = [char2idx.get(ch, 0) for ch in seed_text]

    print("\n>>> Starting End-to-End Training via Custom CUDA Kernels...\n")

    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        step_count = 0
        t0 = time.perf_counter()

        for x_b, y_b in get_batches(encoded, batch_size=batch_size, seq_len=seq_len, device=device):
            loss = model.train_step(x_b, y_b)
            epoch_loss += loss
            step_count += 1
            global_step += 1

            if global_step % 50 == 0:
                t_step = (time.perf_counter() - t0) / step_count
                tokens_per_sec = (seq_len * batch_size) / t_step
                print(
                    f"Epoch [{epoch}/{epochs}] | Step {global_step:4d} | "
                    f"Loss: {loss:.4f} | Perplexity: {math.exp(min(loss, 20)):.2f} | "
                    f"Speed: {tokens_per_sec:,.0f} tokens/s"
                )

        avg_loss = epoch_loss / max(1, step_count)
        print(f"\n[Epoch {epoch} Summary] Average Loss: {avg_loss:.4f} | Elapsed: {time.time() - start_time:.1f}s")

        # Generate sample text
        print("\n--- Model Generation Sample (T=0.8) ---")
        sample_indices = model.generate(seed_indices, length=150, temperature=0.8)
        sample_chars = "".join([idx2char.get(i, "") for i in sample_indices])
        print(sample_chars)
        print("-" * 50 + "\n")

    print("=" * 75)
    print(" [SUCCESS] Training completed successfully with Custom CUDA LSTM! ")
    print("=" * 75)


if __name__ == "__main__":
    main()

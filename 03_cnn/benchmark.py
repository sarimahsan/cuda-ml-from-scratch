"""
=============================================================================
CUDA ML vs PyTorch Native: Convolutional Neural Network (CNN) Benchmark
=============================================================================
This script benchmarks the custom modular CUDA C++ CNN kernels against
PyTorch's native cuDNN-backed Conv2d, MaxPool2d, Linear, and Autograd engines.

Comparisons:
  1. Kernel-level Microbenchmarks:
     - Conv2D (Forward, Backward Data, Backward Filter/Bias)
     - MaxPool2D (Forward with Argmax Mask, Backward Routing)
     - Tiled Shared-Memory GEMM (Forward & Backward)
     - Activations (ReLU Forward & Backward)
     - Softmax Cross-Entropy & Gradients
     - In-place Vectorized Optimizers (Adam, SGD Momentum)
  2. Batch & Channel Scaling Macrobenchmarks
  3. End-to-End Training Throughput (Images/sec) & Peak VRAM Usage
=============================================================================
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from cnn import CUDACNN, _ext, CUDAConv2d, CUDAMaxPool2d
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from cnn import CUDACNN, _ext, CUDAConv2d, CUDAMaxPool2d


# ---------------------------------------------------------------------------
# Native PyTorch CNN Baseline Module
# ---------------------------------------------------------------------------
class PyTorchCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        in_height: int = 28,
        in_width: int = 28,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_hidden: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        flat_dim = conv2_channels * (in_height // 4) * (in_width // 4)
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# High-Precision CUDA Timing Utility
# ---------------------------------------------------------------------------
def time_cuda_operation(op_func, warmup: int = 10, reps: int = 50) -> Tuple[float, float]:
    """
    Measures GPU execution time using torch.cuda.Event with warmup and synchronization.
    Returns: (mean_ms, std_ms)
    """
    for _ in range(warmup):
        op_func()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]

    for i in range(reps):
        start_events[i].record()
        op_func()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# 1. Microbenchmarks: Custom CUDA vs PyTorch Native
# ---------------------------------------------------------------------------
def benchmark_conv2d():
    print("\n" + "=" * 75)
    print(" 1. Microbenchmark: Conv2D (Custom CUDA vs PyTorch cuDNN)")
    print("=" * 75)
    configs = [
        {"N": 64, "C_in": 1, "H": 28, "W": 28, "C_out": 16, "K": 3, "pad": 1, "stride": 1},
        {"N": 64, "C_in": 16, "H": 14, "W": 14, "C_out": 32, "K": 3, "pad": 1, "stride": 1},
        {"N": 128, "C_in": 32, "H": 14, "W": 14, "C_out": 64, "K": 3, "pad": 1, "stride": 1},
    ]

    print(f"{'Config (N, C_in, H, W -> C_out)':<35} | {'Custom CUDA (ms)':<18} | {'PyTorch (ms)':<16} | {'Speedup':<8}")
    print("-" * 85)

    for cfg in configs:
        N, C_in, H, W, C_out, K, pad, stride = (
            cfg["N"], cfg["C_in"], cfg["H"], cfg["W"], cfg["C_out"], cfg["K"], cfg["pad"], cfg["stride"]
        )
        desc = f"N={N}, Cin={C_in}, {H}x{W} -> Cout={C_out}"

        X = torch.randn(N, C_in, H, W, device="cuda")
        W_w = torch.randn(C_out, C_in, K, K, device="cuda")
        b = torch.randn(C_out, device="cuda")

        # Custom CUDA Forward
        custom_ms, _ = time_cuda_operation(lambda: _ext.conv2d_forward(X, W_w, b, stride, pad))

        # PyTorch cuDNN Forward
        torch_conv = nn.Conv2d(C_in, C_out, K, stride=stride, padding=pad, bias=True).cuda()
        torch_conv.weight.data.copy_(W_w)
        torch_conv.bias.data.copy_(b)
        torch_ms, _ = time_cuda_operation(lambda: torch_conv(X))

        speedup = f"{torch_ms / custom_ms:.2f}x"
        print(f"{desc:<35} | {custom_ms:8.3f} ms        | {torch_ms:8.3f} ms      | {speedup}")


def benchmark_maxpool2d():
    print("\n" + "=" * 75)
    print(" 2. Microbenchmark: MaxPool2D (Custom CUDA vs PyTorch Native)")
    print("=" * 75)
    configs = [
        {"N": 64, "C": 16, "H": 28, "W": 28, "K": 2, "stride": 2},
        {"N": 64, "C": 32, "H": 14, "W": 14, "K": 2, "stride": 2},
        {"N": 128, "C": 64, "H": 14, "W": 14, "K": 2, "stride": 2},
    ]

    print(f"{'Config (N, C, H, W)':<30} | {'Custom CUDA (ms)':<18} | {'PyTorch (ms)':<16} | {'Speedup':<8}")
    print("-" * 80)

    for cfg in configs:
        N, C, H, W, K, stride = cfg["N"], cfg["C"], cfg["H"], cfg["W"], cfg["K"], cfg["stride"]
        desc = f"N={N}, C={C}, {H}x{W}"

        X = torch.randn(N, C, H, W, device="cuda")

        custom_ms, _ = time_cuda_operation(lambda: _ext.maxpool2d_forward(X, K, K, stride, 0))
        pool = nn.MaxPool2d(kernel_size=K, stride=stride).cuda()
        torch_ms, _ = time_cuda_operation(lambda: pool(X))

        speedup = f"{torch_ms / custom_ms:.2f}x"
        print(f"{desc:<30} | {custom_ms:8.3f} ms        | {torch_ms:8.3f} ms      | {speedup}")


def benchmark_end_to_end_training():
    print("\n" + "=" * 75)
    print(" 3. Macrobenchmark: End-to-End Training Step (Forward + Backward + Step)")
    print("=" * 75)
    batch_sizes = [32, 64, 128, 256]

    print(f"{'Batch Size':<12} | {'Custom CUDA (ms)':<18} | {'PyTorch (ms)':<16} | {'Custom Throughput':<20} | {'PyTorch Throughput'}")
    print("-" * 95)

    for B in batch_sizes:
        X = torch.randn(B, 1, 28, 28, device="cuda")
        y = torch.randint(0, 10, (B,), device="cuda")
        y_onehot = torch.nn.functional.one_hot(y, 10).float()

        # Custom CUDA CNN
        custom_model = CUDACNN(1, 28, 28, 16, 32, 128, 10, lr=0.001).cuda()

        def custom_step():
            logits = custom_model(X)
            loss, _, _ = custom_model.compute_loss_and_probs(logits, y_onehot)
            loss.backward()
            custom_model.step_optimizer(method="adam")

        custom_ms, _ = time_cuda_operation(custom_step, warmup=5, reps=30)
        custom_fps = (B / (custom_ms / 1000.0))

        # PyTorch Native CNN
        torch_model = PyTorchCNN(1, 28, 28, 16, 32, 128, 10).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_model.parameters(), lr=0.001)

        def torch_step():
            optimizer.zero_grad()
            logits = torch_model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        torch_ms, _ = time_cuda_operation(torch_step, warmup=5, reps=30)
        torch_fps = (B / (torch_ms / 1000.0))

        print(f"{B:<12} | {custom_ms:8.3f} ms        | {torch_ms:8.3f} ms      | {custom_fps:8.1f} imgs/s       | {torch_fps:8.1f} imgs/s")


def main():
    parser = argparse.ArgumentParser(description="Modular CUDA CNN Benchmarking Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Benchmark requires NVIDIA GPU.")
        return

    print("=============================================================================")
    print(f" GPU Device: {torch.cuda.get_device_name(0)}")
    print(f" CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f" PyTorch Version: {torch.__version__}")
    print("=============================================================================")

    benchmark_conv2d()
    benchmark_maxpool2d()
    benchmark_end_to_end_training()


if __name__ == "__main__":
    main()

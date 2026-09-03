"""
=============================================================================
CUDA ML vs PyTorch Native: Logistic Regression Comprehensive Benchmark
=============================================================================
This script benchmarks the custom CUDA C++ Logistic Regression kernels
against PyTorch's native autograd & C10/cuBLAS-backed implementation.

Comparisons:
  1. Microbenchmarks: Forward pass, BCE Loss, Backward Gradients, SGD Step
  2. Macrobenchmarks: End-to-end training epoch scaling across N and D
  3. Numerical Parity: Exact gradient, loss, and weight tolerance verification
  4. Memory Profiling: Peak GPU memory allocation
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

# Import custom CUDA Logistic Regression extension & class
try:
    from logistic_regression import CUDALogisticRegression, _ext
except ImportError:
    # If run from another directory, add module dir to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from logistic_regression import CUDALogisticRegression, _ext


# ---------------------------------------------------------------------------
# Native PyTorch Logistic Regression Reference Model
# ---------------------------------------------------------------------------
class PyTorchLogisticRegression(nn.Module):
    """
    Standard PyTorch Logistic Regression baseline using nn.Linear and Sigmoid.
    """

    def __init__(self, in_features: int, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.linear = nn.Linear(in_features, 1, bias=True, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).view(-1)


# ---------------------------------------------------------------------------
# CUDA Event High-Precision Timing Utilities
# ---------------------------------------------------------------------------
def time_cuda_operation(op_func, warmup: int = 10, reps: int = 50) -> Tuple[float, float]:
    """
    Measures GPU execution time using torch.cuda.Event with warmup and multiple reps.
    Returns:
        (mean_ms, std_ms)
    """
    # Warmup
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
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # in milliseconds
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Benchmark 1: Microbenchmarks (Component-level breakdown)
# ---------------------------------------------------------------------------
def benchmark_micro(N: int = 100_000, D: int = 64, device: str = "cuda") -> List[Dict]:
    """
    Benchmarks individual components: Forward, BCE Loss, Backward Gradients, SGD Update.
    """
    print("=" * 80)
    print(f" 1. MICROBENCHMARK BREAKDOWN: Custom CUDA vs PyTorch Native (N={N:,}, D={D})")
    print("=" * 80)

    torch.manual_seed(42)
    dev = torch.device(device)

    X = torch.randn(N, D, device=dev, dtype=torch.float32)
    y = torch.randint(0, 2, (N,), device=dev, dtype=torch.float32)
    w = torch.randn(D, device=dev, dtype=torch.float32)
    b = torch.randn(1, device=dev, dtype=torch.float32)
    lr = 0.05

    # PyTorch baseline components
    pt_model = PyTorchLogisticRegression(D, device=device)
    with torch.no_grad():
        pt_model.linear.weight.copy_(w.view(1, D))
        pt_model.linear.bias.copy_(b)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(pt_model.parameters(), lr=lr)

    results = []

    # --- A. Forward Pass ---
    def custom_forward():
        return _ext.forward(X, w, b)

    def pytorch_forward():
        return pt_model(X)

    t_cuda_fwd, s_cuda_fwd = time_cuda_operation(custom_forward)
    t_pt_fwd, s_pt_fwd = time_cuda_operation(pytorch_forward)
    results.append({
        "Component": "Forward Pass (z = Xw+b, y_hat = σ(z))",
        "CUDA (ms)": t_cuda_fwd,
        "PyTorch (ms)": t_pt_fwd,
        "Speedup": t_pt_fwd / t_cuda_fwd if t_cuda_fwd > 0 else 0.0,
    })

    # --- B. BCE Loss Reduction ---
    y_hat_cuda = custom_forward()
    y_hat_pt = pytorch_forward()

    def custom_loss():
        return _ext.bce_loss(y_hat_cuda, y)

    def pytorch_loss():
        return criterion(y_hat_pt, y)

    t_cuda_loss, s_cuda_loss = time_cuda_operation(custom_loss)
    t_pt_loss, s_pt_loss = time_cuda_operation(pytorch_loss)
    results.append({
        "Component": "BCE Loss (Warp/Block Reduction)",
        "CUDA (ms)": t_cuda_loss,
        "PyTorch (ms)": t_pt_loss,
        "Speedup": t_pt_loss / t_cuda_loss if t_cuda_loss > 0 else 0.0,
    })

    # --- C. Backward Pass (Gradients) ---
    def custom_backward():
        return _ext.backward(X, y_hat_cuda, y)

    def pytorch_backward():
        pt_model.zero_grad(set_to_none=False)
        out = pt_model(X)
        loss = criterion(out, y)
        loss.backward(retain_graph=True)

    t_cuda_bwd, s_cuda_bwd = time_cuda_operation(custom_backward)
    t_pt_bwd, s_pt_bwd = time_cuda_operation(pytorch_backward)
    results.append({
        "Component": "Backward Pass (dW = X^T(y_hat-y)/N, db)",
        "CUDA (ms)": t_cuda_bwd,
        "PyTorch (ms)": t_pt_bwd,
        "Speedup": t_pt_bwd / t_cuda_bwd if t_cuda_bwd > 0 else 0.0,
    })

    # --- D. Optimizer Update (SGD) ---
    grad_w, grad_b = _ext.backward(X, y_hat_cuda, y)
    w_copy = w.clone()
    b_copy = b.clone()

    def custom_sgd():
        _ext.sgd_step(w_copy, b_copy, grad_w, grad_b, lr)

    def pytorch_sgd():
        optimizer.step()

    t_cuda_sgd, s_cuda_sgd = time_cuda_operation(custom_sgd)
    t_pt_sgd, s_pt_sgd = time_cuda_operation(pytorch_sgd)
    results.append({
        "Component": "Optimizer Step (SGD in-place update)",
        "CUDA (ms)": t_cuda_sgd,
        "PyTorch (ms)": t_pt_sgd,
        "Speedup": t_pt_sgd / t_cuda_sgd if t_cuda_sgd > 0 else 0.0,
    })

    # Print Table
    print(f"{'Component':<45} | {'CUDA (ms)':<10} | {'PyTorch (ms)':<12} | {'Speedup':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['Component']:<45} | {r['CUDA (ms)']:>10.4f} | {r['PyTorch (ms)']:>12.4f} | {r['Speedup']:>7.2f}x")
    print("-" * 80)

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Macrobenchmarks (Scaling Dataset Size & Features)
# ---------------------------------------------------------------------------
def benchmark_macro_scaling(device: str = "cuda") -> Tuple[List[Dict], List[Dict]]:
    """
    Benchmarks full training step latency & throughput across:
      A. Scaling Sample Count N (10k, 50k, 100k, 500k, 1M) with fixed D=64
      B. Scaling Feature Count D (16, 64, 256, 1024) with fixed N=100k
    """
    print("\n" + "=" * 85)
    print(" 2. MACROBENCHMARK SCALING: Full Training Step (Forward + Loss + Backward + SGD)")
    print("=" * 85)

    dev = torch.device(device)
    lr = 0.05
    n_configs = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    d_fixed = 64

    results_n = []

    print(f"\n--- [A] Scaling Dataset Size N (Fixed D = {d_fixed}) ---")
    print(f"{'N Samples':<12} | {'CUDA (ms)':<10} | {'PyTorch (ms)':<12} | {'CUDA Thrpt (M/s)':<17} | {'Speedup':<8}")
    print("-" * 75)

    for N in n_configs:
        X = torch.randn(N, d_fixed, device=dev, dtype=torch.float32)
        y = torch.randint(0, 2, (N,), device=dev, dtype=torch.float32)
        w = torch.zeros(d_fixed, device=dev, dtype=torch.float32)
        b = torch.zeros(1, device=dev, dtype=torch.float32)

        pt_model = PyTorchLogisticRegression(d_fixed, device=device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(pt_model.parameters(), lr=lr)

        def run_cuda_full_step():
            y_hat = _ext.forward(X, w, b)
            loss = _ext.bce_loss(y_hat, y)
            gw, gb = _ext.backward(X, y_hat, y)
            _ext.sgd_step(w, b, gw, gb, lr)

        def run_pt_full_step():
            optimizer.zero_grad(set_to_none=True)
            y_hat = pt_model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        t_cuda, _ = time_cuda_operation(run_cuda_full_step, reps=30)
        t_pt, _ = time_cuda_operation(run_pt_full_step, reps=30)
        speedup = t_pt / t_cuda if t_cuda > 0 else 0.0
        throughput_cuda = (N / (t_cuda / 1000.0)) / 1e6  # Million samples/sec

        results_n.append({
            "N": N,
            "D": d_fixed,
            "CUDA (ms)": t_cuda,
            "PyTorch (ms)": t_pt,
            "Throughput (M/s)": throughput_cuda,
            "Speedup": speedup,
        })

        print(f"{N:<12,d} | {t_cuda:>10.4f} | {t_pt:>12.4f} | {throughput_cuda:>17.2f} | {speedup:>7.2f}x")

    # --- [B] Scaling Feature Dimension D ---
    n_fixed = 100_000
    d_configs = [16, 64, 256, 1024]
    results_d = []

    print(f"\n--- [B] Scaling Feature Count D (Fixed N = {n_fixed:,}) ---")
    print(f"{'D Features':<12} | {'CUDA (ms)':<10} | {'PyTorch (ms)':<12} | {'CUDA Thrpt (M/s)':<17} | {'Speedup':<8}")
    print("-" * 75)

    for D in d_configs:
        X = torch.randn(n_fixed, D, device=dev, dtype=torch.float32)
        y = torch.randint(0, 2, (n_fixed,), device=dev, dtype=torch.float32)
        w = torch.zeros(D, device=dev, dtype=torch.float32)
        b = torch.zeros(1, device=dev, dtype=torch.float32)

        pt_model = PyTorchLogisticRegression(D, device=device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(pt_model.parameters(), lr=lr)

        def run_cuda_full_step_d():
            y_hat = _ext.forward(X, w, b)
            loss = _ext.bce_loss(y_hat, y)
            gw, gb = _ext.backward(X, y_hat, y)
            _ext.sgd_step(w, b, gw, gb, lr)

        def run_pt_full_step_d():
            optimizer.zero_grad(set_to_none=True)
            y_hat = pt_model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        t_cuda, _ = time_cuda_operation(run_cuda_full_step_d, reps=30)
        t_pt, _ = time_cuda_operation(run_pt_full_step_d, reps=30)
        speedup = t_pt / t_cuda if t_cuda > 0 else 0.0
        throughput_cuda = (n_fixed / (t_cuda / 1000.0)) / 1e6

        results_d.append({
            "N": n_fixed,
            "D": D,
            "CUDA (ms)": t_cuda,
            "PyTorch (ms)": t_pt,
            "Throughput (M/s)": throughput_cuda,
            "Speedup": speedup,
        })

        print(f"{D:<12d} | {t_cuda:>10.4f} | {t_pt:>12.4f} | {throughput_cuda:>17.2f} | {speedup:>7.2f}x")

    return results_n, results_d


# ---------------------------------------------------------------------------
# Benchmark 3: Numerical Parity & Mathematical Convergence Verification
# ---------------------------------------------------------------------------
def verify_numerical_parity(N: int = 10_000, D: int = 32, num_steps: int = 50, device: str = "cuda"):
    """
    Verifies that custom CUDA forward hypothesis, BCE loss, analytical gradients,
    and parameter trajectories match PyTorch native autograd within numerical tolerance.
    """
    print("\n" + "=" * 80)
    print(f" 3. NUMERICAL PARITY & CONVERGENCE VALIDATION (N={N:,}, D={D}, Steps={num_steps})")
    print("=" * 80)

    torch.manual_seed(1337)
    dev = torch.device(device)

    X = torch.randn(N, D, device=dev, dtype=torch.float32)
    # Synthetic target with ground-truth weights
    true_w = torch.randn(D, device=dev)
    true_b = 0.5
    prob = torch.sigmoid(X @ true_w + true_b)
    y = (prob > 0.5).float()

    lr = 0.1

    # Initial weights
    w_cuda = torch.zeros(D, device=dev, dtype=torch.float32)
    b_cuda = torch.zeros(1, device=dev, dtype=torch.float32)

    pt_model = PyTorchLogisticRegression(D, device=device)
    with torch.no_grad():
        pt_model.linear.weight.zero_()
        pt_model.linear.bias.zero_()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(pt_model.parameters(), lr=lr)

    # 1. Check Step 1 Forward & Loss Parity
    y_hat_cuda = _ext.forward(X, w_cuda, b_cuda)
    y_hat_pt = pt_model(X)
    fwd_diff = torch.max(torch.abs(y_hat_cuda - y_hat_pt)).item()

    loss_cuda = _ext.bce_loss(y_hat_cuda, y).item()
    loss_pt = criterion(y_hat_pt, y).item()
    loss_diff = abs(loss_cuda - loss_pt)

    # 2. Check Step 1 Gradient Parity
    gw_cuda, gb_cuda = _ext.backward(X, y_hat_cuda, y)
    loss_pt_t = criterion(y_hat_pt, y)
    loss_pt_t.backward()

    gw_pt = pt_model.linear.weight.grad.view(-1)
    gb_pt = pt_model.linear.bias.grad.view(-1)

    grad_w_diff = torch.max(torch.abs(gw_cuda - gw_pt)).item()
    grad_b_diff = torch.max(torch.abs(gb_cuda - gb_pt)).item()

    print(f"• Forward Predictions Max Abs Diff : {fwd_diff:12.6e} (Threshold: < 1e-4) -> {'[PASSED]' if fwd_diff < 1e-4 else '[FAILED]'}")
    print(f"• BCE Loss Absolute Difference     : {loss_diff:12.6e} (Threshold: < 1e-4) -> {'[PASSED]' if loss_diff < 1e-4 else '[FAILED]'}")
    print(f"• Weight Gradients Max Abs Diff    : {grad_w_diff:12.6e} (Threshold: < 1e-4) -> {'[PASSED]' if grad_w_diff < 1e-4 else '[FAILED]'}")
    print(f"• Bias Gradient Max Abs Diff      : {grad_b_diff:12.6e} (Threshold: < 1e-4) -> {'[PASSED]' if grad_b_diff < 1e-4 else '[FAILED]'}")

    # 3. Multi-step Training Trajectory Comparison
    losses_cuda = []
    losses_pt = []

    for step in range(num_steps):
        # CUDA Step
        y_hat_c = _ext.forward(X, w_cuda, b_cuda)
        l_c = _ext.bce_loss(y_hat_c, y).item()
        gw_c, gb_c = _ext.backward(X, y_hat_c, y)
        _ext.sgd_step(w_cuda, b_cuda, gw_c, gb_c, lr)
        losses_cuda.append(l_c)

        # PyTorch Step
        optimizer.zero_grad(set_to_none=True)
        y_hat_p = pt_model(X)
        l_p = criterion(y_hat_p, y)
        l_p.backward()
        optimizer.step()
        losses_pt.append(l_p.item())

    final_w_diff = torch.max(torch.abs(w_cuda - pt_model.linear.weight.view(-1))).item()
    final_b_diff = torch.max(torch.abs(b_cuda - pt_model.linear.bias.view(-1))).item()
    final_loss_diff = abs(losses_cuda[-1] - losses_pt[-1])

    print(f"• Final Loss after {num_steps} Steps    : CUDA={losses_cuda[-1]:.6f} | PyTorch={losses_pt[-1]:.6f} (Diff: {final_loss_diff:8.2e})")
    print(f"• Final Weight Parameters Max Diff : {final_w_diff:12.6e} -> {'[PASSED]' if final_w_diff < 1e-3 else '[FAILED]'}")
    print(f"• Final Bias Parameter Max Diff    : {final_b_diff:12.6e} -> {'[PASSED]' if final_b_diff < 1e-3 else '[FAILED]'}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Benchmark 4: VRAM Memory Profiling
# ---------------------------------------------------------------------------
def benchmark_memory(N: int = 1_000_000, D: int = 128, device: str = "cuda"):
    """
    Profiles peak GPU memory consumption during training steps.
    """
    print("\n" + "=" * 80)
    print(f" 4. PEAK GPU MEMORY CONSUMPTION PROFILE (N={N:,}, D={D})")
    print("=" * 80)

    dev = torch.device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    # 1. Custom CUDA Step Memory
    X = torch.randn(N, D, device=dev, dtype=torch.float32)
    y = torch.randint(0, 2, (N,), device=dev, dtype=torch.float32)
    w = torch.zeros(D, device=dev, dtype=torch.float32)
    b = torch.zeros(1, device=dev, dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats(dev)
    y_hat = _ext.forward(X, w, b)
    loss = _ext.bce_loss(y_hat, y)
    gw, gb = _ext.backward(X, y_hat, y)
    _ext.sgd_step(w, b, gw, gb, 0.01)
    torch.cuda.synchronize()
    cuda_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    # Clean up
    del X, y, w, b, y_hat, loss, gw, gb
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    # 2. PyTorch Native Step Memory
    X_pt = torch.randn(N, D, device=dev, dtype=torch.float32)
    y_pt = torch.randint(0, 2, (N,), device=dev, dtype=torch.float32)
    pt_model = PyTorchLogisticRegression(D, device=device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(pt_model.parameters(), lr=0.01)

    torch.cuda.reset_peak_memory_stats(dev)
    optimizer.zero_grad(set_to_none=True)
    out = pt_model(X_pt)
    loss_pt = criterion(out, y_pt)
    loss_pt.backward()
    optimizer.step()
    torch.cuda.synchronize()
    pt_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    print(f"• Dataset Memory (N={N:,}, D={D}, float32) : {(N * D * 4) / (1024 * 1024):.2f} MB")
    print(f"• Custom CUDA Kernel Peak VRAM Allocation   : {cuda_peak_mb:.2f} MB")
    print(f"• PyTorch Native Autograd Peak VRAM         : {pt_peak_mb:.2f} MB")
    print(f"• Memory Overhead Difference                : {abs(pt_peak_mb - cuda_peak_mb):.2f} MB")
    print("=" * 80)


def init_and_warmup_cuda(device: str = "cuda"):
    """
    Explicitly initializes CUDA runtime, primary context, and PyTorch's internal cuBLAS / Autograd handle pool.
    Runs warmup kernel & backward passes to prevent lazy-initialization jitter from contaminating microbenchmarks.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return
    torch.cuda.init()
    torch.cuda.synchronize()

    dummy_x = torch.randn(256, 64, device=device, requires_grad=True)
    dummy_w = torch.randn(64, 1, device=device, requires_grad=True)
    for _ in range(50):
        dummy_y = torch.sigmoid(torch.matmul(dummy_x, dummy_w))
        dummy_loss = dummy_y.sum()
        dummy_loss.backward()
        torch.cuda.synchronize()

    del dummy_x, dummy_w, dummy_y, dummy_loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CUDA ML vs PyTorch Logistic Regression Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--quick", action="store_true", help="Run a quick version of benchmarks")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Please run on a GPU-enabled runtime (e.g. Google Colab).")
        sys.exit(1)

    # Initialize CUDA context and warm up cuBLAS before timing
    init_and_warmup_cuda(device=args.device)

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 80)
    print("      🚀 CUDA ML BENCHMARK SUITE: BINARY LOGISTIC REGRESSION 🚀")
    print("=" * 80)
    print(f"• GPU Device         : {device_name}")
    print(f"• Compute Capability : {device_props.major}.{device_props.minor}")
    print(f"• Total VRAM         : {device_props.total_memory / (1024**3):.2f} GB")
    print(f"• PyTorch Version    : {torch.__version__}")
    print(f"• CUDA Version       : {torch.version.cuda}")
    print("=" * 80 + "\n")

    # 1. Microbenchmarks
    benchmark_micro(N=50_000 if args.quick else 100_000, D=64, device=args.device)

    # 2. Macro Scaling
    benchmark_macro_scaling(device=args.device)

    # 3. Numerical Parity
    verify_numerical_parity(N=20_000 if args.quick else 50_000, D=32, num_steps=30, device=args.device)

    # 4. Memory Profiling
    benchmark_memory(N=500_000 if args.quick else 1_000_000, D=64, device=args.device)

    print("\n[SUCCESS] Logistic Regression Benchmarking Complete!\n")


if __name__ == "__main__":
    main()

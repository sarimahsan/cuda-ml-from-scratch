"""
=============================================================================
CUDA ML BENCHMARK SUITE: GATED RECURRENT UNIT (GRU)
=============================================================================
Comprehensive correctness verification and micro/macro throughput benchmarks:
  1. Forward & Backward (BPTT) Numerical Gradient Checking vs torch.nn.GRU
  2. Microbenchmark: Step & Sequence Operations vs Native PyTorch cuDNN
  3. Macrobenchmark: Sequence Length & Batch Size Throughput Scaling
=============================================================================
"""

import argparse
import sys
from typing import Callable, Tuple
import torch
import torch.nn as nn

try:
    from gru import CUDAGRU, CUDAGRUCell, _ext
except ImportError as e:
    print(f"[ERROR] Failed to import CUDA GRU: {e}")
    sys.exit(1)


def time_cuda_operation(func: Callable, warmup: int = 25, reps: int = 100) -> Tuple[float, float]:
    """Measures execution time strictly using device-side CUDA events."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(reps):
        func()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / reps
    return elapsed_ms, 0.0


def test_correctness(device: str = "cuda"):
    print("\n" + "=" * 85)
    print(" 1. NUMERICAL CORRECTNESS & GRADIENT VERIFICATION (vs torch.nn.GRU)")
    print("=" * 85)

    T, N, D, H = 16, 8, 32, 64
    torch.manual_seed(42)

    # 1. Instantiate PyTorch Native and Custom CUDA GRU
    gru_torch = nn.GRU(input_size=D, hidden_size=H, bias=True, batch_first=False).to(device)
    gru_cuda = CUDAGRU(input_size=D, hidden_size=H, bias=True).to(device)

    # Sync weights
    with torch.no_grad():
        gru_cuda.weight_ih.copy_(gru_torch.weight_ih_l0.t())
        gru_cuda.weight_hh.copy_(gru_torch.weight_hh_l0.t())
        gru_cuda.bias_ih.copy_(gru_torch.bias_ih_l0)
        gru_cuda.bias_hh.copy_(gru_torch.bias_hh_l0)

    # Inputs
    X_torch = torch.randn(T, N, D, device=device, requires_grad=True)
    X_cuda = X_torch.clone().detach().requires_grad_(True)
    h0_torch = torch.randn(1, N, H, device=device, requires_grad=True)
    h0_cuda = h0_torch.squeeze(0).clone().detach()

    # Forward
    out_torch, hn_torch = gru_torch(X_torch, h0_torch)
    out_cuda, hn_cuda = gru_cuda(X_cuda, h0_cuda)

    max_fwd_diff = (out_torch - out_cuda).abs().max().item()
    max_hn_diff = (hn_torch.squeeze(0) - hn_cuda).abs().max().item()

    # Backward
    loss_torch = out_torch.sum()
    loss_torch.backward()

    loss_cuda = out_cuda.sum()
    loss_cuda.backward()

    max_dx_diff = (X_torch.grad - X_cuda.grad).abs().max().item()
    max_dw_ih_diff = (gru_torch.weight_ih_l0.grad - gru_cuda.weight_ih.grad.t()).abs().max().item()
    max_dw_hh_diff = (gru_torch.weight_hh_l0.grad - gru_cuda.weight_hh.grad.t()).abs().max().item()
    max_db_ih_diff = (gru_torch.bias_ih_l0.grad - gru_cuda.bias_ih.grad).abs().max().item()
    max_db_hh_diff = (gru_torch.bias_hh_l0.grad - gru_cuda.bias_hh.grad).abs().max().item()

    print(f" • Forward Output Max Difference   : {max_fwd_diff:.2e}  -> {'✅ PASS' if max_fwd_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Final State Max Difference      : {max_hn_diff:.2e}  -> {'✅ PASS' if max_hn_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Input Gradient (dX) Difference  : {max_dx_diff:.2e}  -> {'✅ PASS' if max_dx_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Weight IH Gradient Difference   : {max_dw_ih_diff:.2e}  -> {'✅ PASS' if max_dw_ih_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Weight HH Gradient Difference   : {max_dw_hh_diff:.2e}  -> {'✅ PASS' if max_dw_hh_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Bias IH Gradient Difference     : {max_db_ih_diff:.2e}  -> {'✅ PASS' if max_db_ih_diff < 1e-4 else '❌ FAIL'}")
    print(f" • Bias HH Gradient Difference     : {max_db_hh_diff:.2e}  -> {'✅ PASS' if max_db_hh_diff < 1e-4 else '❌ FAIL'}")

    assert max_fwd_diff < 1e-3, "Forward verification failed!"
    assert max_dx_diff < 1e-3, "Backward verification failed!"


def benchmark_micro(device: str = "cuda"):
    print("\n" + "=" * 85)
    print(" 2. KERNEL-LEVEL MICROBENCHMARKS: Custom CUDA GRU vs PyTorch Native cuDNN")
    print("    Dimensions: Sequence T=64, Batch N=64, Input D=128, Hidden H=256")
    print("=" * 85)
    print(f"{'Operation':<42} | {'CUDA (ms)':<11} | {'PyTorch (ms)':<12} | {'Speedup':<8}")
    print("-" * 85)

    T, N, D, H = 64, 64, 128, 256
    X = torch.randn(T, N, D, device=device)
    h0 = torch.zeros(N, H, device=device)
    h0_torch = torch.zeros(1, N, H, device=device)

    gru_torch = nn.GRU(D, H, bias=True, batch_first=False).to(device)
    gru_cuda = CUDAGRU(D, H, bias=True).to(device)

    # Forward
    t_fwd_cuda, _ = time_cuda_operation(lambda: gru_cuda(X, h0))
    t_fwd_torch, _ = time_cuda_operation(lambda: gru_torch(X, h0_torch))
    sp_fwd = t_fwd_torch / max(t_fwd_cuda, 1e-6)
    print(f"{'GRU Sequence Forward (T=64, N=64)':<42} | {t_fwd_cuda:>9.4f} ms | {t_fwd_torch:>10.4f} ms | {sp_fwd:>6.2f}x")

    # Forward + Backward
    def run_cuda_full():
        x_c = X.clone().detach().requires_grad_(True)
        out, _ = gru_cuda(x_c, h0)
        out.sum().backward()

    def run_torch_full():
        x_t = X.clone().detach().requires_grad_(True)
        out, _ = gru_torch(x_t, h0_torch)
        out.sum().backward()

    t_full_cuda, _ = time_cuda_operation(run_cuda_full, warmup=15, reps=50)
    t_full_torch, _ = time_cuda_operation(run_torch_full, warmup=15, reps=50)
    sp_full = t_full_torch / max(t_full_cuda, 1e-6)
    print(f"{'GRU Full Forward + BPTT (T=64, N=64)':<42} | {t_full_cuda:>9.4f} ms | {t_full_torch:>10.4f} ms | {sp_full:>6.2f}x")
    print("-" * 85)


def benchmark_macro(device: str = "cuda"):
    print("\n" + "=" * 90)
    print(" 3. MACROBENCHMARK: Sequence Length & Batch Size Scaling Throughput")
    print("=" * 90)
    print(f"{'Config (T x N x D x H)':<24} | {'CUDA (ms)':<11} | {'PyTorch (ms)':<12} | {'CUDA Thrpt (tok/s)':<18} | {'Speedup':<8}")
    print("-" * 90)

    configs = [
        (32, 32, 128, 128),
        (64, 64, 128, 256),
        (128, 64, 256, 512),
        (128, 128, 256, 512),
        (256, 64, 256, 512),
    ]

    for T, N, D, H in configs:
        X = torch.randn(T, N, D, device=device)
        h0 = torch.zeros(N, H, device=device)
        h0_torch = torch.zeros(1, N, H, device=device)

        gru_torch = nn.GRU(D, H, bias=True, batch_first=False).to(device)
        gru_cuda = CUDAGRU(D, H, bias=True).to(device)

        def run_cuda():
            x = X.clone().detach().requires_grad_(True)
            out, _ = gru_cuda(x, h0)
            out.sum().backward()

        def run_torch():
            x = X.clone().detach().requires_grad_(True)
            out, _ = gru_torch(x, h0_torch)
            out.sum().backward()

        t_cuda, _ = time_cuda_operation(run_cuda, warmup=10, reps=30)
        t_torch, _ = time_cuda_operation(run_torch, warmup=10, reps=30)

        total_tokens = T * N
        throughput = total_tokens / (t_cuda / 1000.0)
        speedup = t_torch / max(t_cuda, 1e-6)

        cfg_str = f"T={T}, N={N}, H={H}"
        print(f"{cfg_str:<24} | {t_cuda:>9.4f} ms | {t_torch:>10.4f} ms | {throughput:>16,.0f} | {speedup:>6.2f}x")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="CUDA ML Benchmark Suite: GRU")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to execute this benchmark suite.")
        sys.exit(1)

    print("=" * 85)
    print(" 🚀 CUDA ML BENCHMARK SUITE: GATED RECURRENT UNIT (GRU) 🚀")
    print(f" • GPU Device         : {torch.cuda.get_device_name(0)}")
    print(f" • PyTorch Version    : {torch.__version__}")
    print(f" • CUDA Device Count  : {torch.cuda.device_count()}")
    print("=" * 85)

    test_correctness(device=args.device)
    benchmark_micro(device=args.device)
    benchmark_macro(device=args.device)


if __name__ == "__main__":
    main()

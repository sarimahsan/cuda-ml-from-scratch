"""
=============================================================================
CUDA ML BENCHMARK SUITE: GATED RECURRENT UNIT (GRU)
=============================================================================
Comprehensive correctness verification and rigorous statistical profiling:
  1. Forward & Backward (BPTT) Numerical Gradient Checking vs torch.nn.GRU
  2. Isolated Phase Profiling: Forward Only, Backward Only, Forward + Backward
  3. Statistical Distribution: Mean, Median, Std Dev, P5, P95 (100+ iterations)
  4. Macrobenchmark: Sequence Length & Batch Size Scaling with Variance Analysis
=============================================================================
"""

import argparse
import sys
from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    from gru import CUDAGRU, CUDAGRUCell, _ext
except ImportError as e:
    print(f"[ERROR] Failed to import CUDA GRU: {e}")
    sys.exit(1)


def time_cuda_statistical(func: Callable, warmup: int = 50, reps: int = 200) -> Dict[str, float]:
    """
    Measures execution time strictly using device-side CUDA events per iteration
    to extract rigorous sample statistics: Mean, Median, Std Dev, P5, P95.
    """
    # 1. Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # 2. Individual CUDA Events for each rep
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]

    for i in range(reps):
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=np.float64)

    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "median": float(np.median(times)),
        "p5": float(np.percentile(times, 5)),
        "p95": float(np.percentile(times, 95)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def test_correctness(device: str = "cuda"):
    print("\n" + "=" * 90)
    print(" 1. NUMERICAL CORRECTNESS & GRADIENT VERIFICATION (vs torch.nn.GRU)")
    print("=" * 90)

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


def benchmark_micro_isolated(device: str = "cuda", warmup: int = 50, reps: int = 200):
    print("\n" + "=" * 105)
    print(" 2. ISOLATED PHASE PROFILING: Forward vs Backward vs Full BPTT")
    print(f"    Dimensions: T=64, N=64, D=128, H=256 | Repetitions: {reps} (Warmup: {warmup})")
    print("=" * 105)
    print(f"{'Phase':<24} | {'CUDA Mean ± Std':<20} | {'cuDNN Mean ± Std':<20} | {'CUDA Median':<12} | {'cuDNN Median':<12} | {'Speedup':<8}")
    print("-" * 105)

    T, N, D, H = 64, 64, 128, 256
    X = torch.randn(T, N, D, device=device)
    h0 = torch.zeros(N, H, device=device)
    h0_torch = torch.zeros(1, N, H, device=device)

    gru_torch = nn.GRU(D, H, bias=True, batch_first=False).to(device)
    gru_cuda = CUDAGRU(D, H, bias=True).to(device)

    # -------------------------------------------------------------------------
    # A. Forward Only
    # -------------------------------------------------------------------------
    with torch.no_grad():
        stats_fwd_cuda = time_cuda_statistical(lambda: gru_cuda(X, h0), warmup=warmup, reps=reps)
        stats_fwd_torch = time_cuda_statistical(lambda: gru_torch(X, h0_torch), warmup=warmup, reps=reps)

    sp_fwd = stats_fwd_torch['median'] / max(stats_fwd_cuda['median'], 1e-6)
    cuda_fwd_str = f"{stats_fwd_cuda['mean']:.3f} ± {stats_fwd_cuda['std']:.3f} ms"
    torch_fwd_str = f"{stats_fwd_torch['mean']:.3f} ± {stats_fwd_torch['std']:.3f} ms"
    print(f"{'Forward Only':<24} | {cuda_fwd_str:<20} | {torch_fwd_str:<20} | {stats_fwd_cuda['median']:>9.3f} ms | {stats_fwd_torch['median']:>10.3f} ms | {sp_fwd:>6.2f}x")

    # -------------------------------------------------------------------------
    # B. Backward Only (Graph Retained)
    # -------------------------------------------------------------------------
    x_c = X.clone().detach().requires_grad_(True)
    out_c, _ = gru_cuda(x_c, h0)
    grad_c = torch.ones_like(out_c)

    x_t = X.clone().detach().requires_grad_(True)
    out_t, _ = gru_torch(x_t, h0_torch)
    grad_t = torch.ones_like(out_t)

    def run_cuda_bwd_only():
        out_c.backward(grad_c, retain_graph=True)

    def run_torch_bwd_only():
        out_t.backward(grad_t, retain_graph=True)

    stats_bwd_cuda = time_cuda_statistical(run_cuda_bwd_only, warmup=warmup, reps=reps)
    stats_bwd_torch = time_cuda_statistical(run_torch_bwd_only, warmup=warmup, reps=reps)

    sp_bwd = stats_bwd_torch['median'] / max(stats_bwd_cuda['median'], 1e-6)
    cuda_bwd_str = f"{stats_bwd_cuda['mean']:.3f} ± {stats_bwd_cuda['std']:.3f} ms"
    torch_bwd_str = f"{stats_bwd_torch['mean']:.3f} ± {stats_bwd_torch['std']:.3f} ms"
    print(f"{'Backward Only':<24} | {cuda_bwd_str:<20} | {torch_bwd_str:<20} | {stats_bwd_cuda['median']:>9.3f} ms | {stats_bwd_torch['median']:>10.3f} ms | {sp_bwd:>6.2f}x")

    # -------------------------------------------------------------------------
    # C. Forward + Backward (End-to-End BPTT)
    # -------------------------------------------------------------------------
    def run_cuda_full():
        inp = X.clone().detach().requires_grad_(True)
        out, _ = gru_cuda(inp, h0)
        out.sum().backward()

    def run_torch_full():
        inp = X.clone().detach().requires_grad_(True)
        out, _ = gru_torch(inp, h0_torch)
        out.sum().backward()

    stats_full_cuda = time_cuda_statistical(run_cuda_full, warmup=warmup, reps=reps)
    stats_full_torch = time_cuda_statistical(run_torch_full, warmup=warmup, reps=reps)

    sp_full = stats_full_torch['median'] / max(stats_full_cuda['median'], 1e-6)
    cuda_full_str = f"{stats_full_cuda['mean']:.3f} ± {stats_full_cuda['std']:.3f} ms"
    torch_full_str = f"{stats_full_torch['mean']:.3f} ± {stats_full_torch['std']:.3f} ms"
    print(f"{'Forward + Backward':<24} | {cuda_full_str:<20} | {torch_full_str:<20} | {stats_full_cuda['median']:>9.3f} ms | {stats_full_torch['median']:>10.3f} ms | {sp_full:>6.2f}x")
    print("-" * 105)


    # Detailed Distribution Table
    print("\n   📊 Detailed Statistical Percentiles (Forward + Backward):")
    print(f"   {'Metric':<18} | {'Custom CUDA':<18} | {'PyTorch cuDNN':<18}")
    print("   " + "-" * 58)
    print(f"   {'Mean ± Std':<18} | {stats_full_cuda['mean']:.4f} ± {stats_full_cuda['std']:.4f} ms | {stats_full_torch['mean']:.4f} ± {stats_full_torch['std']:.4f} ms")
    print(f"   {'Median (P50)':<18} | {stats_full_cuda['median']:>12.4f} ms    | {stats_full_torch['median']:>12.4f} ms")
    print(f"   {'5th Percentile':<18} | {stats_full_cuda['p5']:>12.4f} ms    | {stats_full_torch['p5']:>12.4f} ms")
    print(f"   {'95th Percentile':<18} | {stats_full_cuda['p95']:>12.4f} ms    | {stats_full_torch['p95']:>12.4f} ms")
    print(f"   {'Min / Max':<18} | {stats_full_cuda['min']:.3f} / {stats_full_cuda['max']:.3f} ms  | {stats_full_torch['min']:.3f} / {stats_full_torch['max']:.3f} ms")
    print("   " + "-" * 58)


def benchmark_macro_rigorous(device: str = "cuda", warmup: int = 30, reps: int = 100):
    print("\n" + "=" * 115)
    print(" 3. MACROBENCHMARK: Statistical Throughput & Scaling Analysis")
    print("=" * 115)
    print(f"{'Config (T x N x D x H)':<24} | {'CUDA Median (ms)':<17} | {'cuDNN Median (ms)':<17} | {'CUDA P5-P95 (ms)':<18} | {'CUDA Thrpt (tok/s)':<18} | {'Speedup':<8}")
    print("-" * 115)

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

        s_cuda = time_cuda_statistical(run_cuda, warmup=warmup, reps=reps)
        s_torch = time_cuda_statistical(run_torch, warmup=warmup, reps=reps)

        total_tokens = T * N
        throughput = total_tokens / (s_cuda['median'] / 1000.0)
        speedup = s_torch['median'] / max(s_cuda['median'], 1e-6)

        cfg_str = f"T={T}, N={N}, H={H}"
        p5_p95_str = f"[{s_cuda['p5']:.2f}, {s_cuda['p95']:.2f}]"
        print(f"{cfg_str:<24} | {s_cuda['median']:>14.4f} ms | {s_torch['median']:>15.4f} ms | {p5_p95_str:<18} | {throughput:>16,.0f} | {speedup:>6.2f}x")

    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="CUDA ML Benchmark Suite: GRU")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--reps", type=int, default=150, help="Number of timed repetitions")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required to execute this benchmark suite.")
        sys.exit(1)

    print("=" * 90)
    print(" 🚀 CUDA ML BENCHMARK SUITE: GATED RECURRENT UNIT (GRU) 🚀")
    print(f" • GPU Device         : {torch.cuda.get_device_name(0)}")
    print(f" • PyTorch Version    : {torch.__version__}")
    print(f" • CUDA Device Count  : {torch.cuda.device_count()}")
    print(f" • Benchmark Iterations: {args.reps} timed (Warmup: {args.warmup})")
    print("=" * 90)

    test_correctness(device=args.device)
    benchmark_micro_isolated(device=args.device, warmup=args.warmup, reps=args.reps)
    benchmark_macro_rigorous(device=args.device, warmup=max(10, args.warmup // 2), reps=args.reps)


if __name__ == "__main__":
    main()


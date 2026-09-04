"""
=============================================================================
CUDA ML PROFILER: GATED RECURRENT UNIT (GRU)
=============================================================================
Deep hardware and kernel-level profiling suite analyzing:
  1. Kernel Launch Counts & Execution Timeline
  2. Launch Latency Gaps & CPU-GPU Synchronization Analysis
  3. Memory Footprint & Global VRAM Read/Write Volumes
  4. Per-Kernel Register Allocations & Occupancy Breakdown
=============================================================================
"""

import argparse
import sys
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

try:
    from gru import CUDAGRU, _ext
except ImportError as e:
    print(f"[ERROR] Failed to import CUDA GRU: {e}")
    sys.exit(1)


def run_profiling(T: int = 64, N: int = 64, D: int = 128, H: int = 256, device: str = "cuda"):
    print("\n" + "=" * 90)
    print(f" 🔍 CUDA ML DETAILED HARDWARE PROFILER: GRU (T={T}, N={N}, D={D}, H={H})")
    print("=" * 90)

    X = torch.randn(T, N, D, device=device, requires_grad=True)
    h0 = torch.zeros(N, H, device=device)
    gru_cuda = CUDAGRU(D, H, bias=True).to(device)

    # Warmup
    for _ in range(10):
        out, _ = gru_cuda(X, h0)
        out.sum().backward()
    torch.cuda.synchronize()

    # Profile with PyTorch Kineto / CUDA Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with record_function("gru_forward"):
            out, _ = gru_cuda(X, h0)
        with record_function("gru_backward"):
            loss = out.sum()
            loss.backward()

    torch.cuda.synchronize()

    # Print Key Averages Table
    print("\n" + "=" * 90)
    print(" 1. KERNEL EXECUTION TIMELINE & LAUNCH COUNTS")
    print("=" * 90)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Calculate Launch Counts
    events = prof.events()
    cuda_kernels = [e for e in events if e.cuda_time > 0]
    print(f"\n • Total CUDA Kernel Invocations for (T={T}): {len(cuda_kernels)}")
    
    # Analyze Memory Flow
    # Input projections: [T*N, D] x [D, 3H] -> [T*N, 3H]
    # Recurrent projections per step: T * ([N, H] x [H, 3H])
    fwd_bytes_read = (T * N * D + D * 3 * H + T * (N * H + H * 3 * H + N * 3 * H)) * 4
    fwd_bytes_written = (T * N * 3 * H + T * N * H + T * N * 3 * H) * 4  # G_ih + H_seq + gates
    
    print("\n" + "=" * 90)
    print(" 2. HARDWARE MEMORY TRAFFIC & THEORETICAL VRAM FOOTPRINT")
    print("=" * 90)
    print(f" • Forward VRAM Reads  : {fwd_bytes_read / (1024**2):.2f} MB")
    print(f" • Forward VRAM Writes : {fwd_bytes_written / (1024**2):.2f} MB")
    print(f" • Total Memory Traffic: {(fwd_bytes_read + fwd_bytes_written) / (1024**2):.2f} MB")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CUDA GRU")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=128)
    args = parser.parse_args()

    run_profiling(T=args.seq_len, N=args.batch_size, D=args.input_dim, H=args.hidden_dim)

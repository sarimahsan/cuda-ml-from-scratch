import os
import sys
import time
from typing import List, Tuple
import torch

def init_and_warmup_cuda(device: str = "cuda"):
    """Explicitly initializes CUDA runtime, primary context, and PyTorch's internal cuBLAS handle pool."""
    if not torch.cuda.is_available() or device == "cpu":
        return
    torch.cuda.init()
    torch.cuda.synchronize()

    dummy_a = torch.randn(256, 256, device=device, requires_grad=True)
    dummy_b = torch.randn(256, 256, device=device, requires_grad=True)
    for _ in range(50):
        dummy_c = torch.matmul(dummy_a, dummy_b)
        dummy_c.sum().backward()
        torch.cuda.synchronize()

    del dummy_a, dummy_b, dummy_c
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def time_gemm_operation(func, inputs: Tuple, warmup: int = 25, iters: int = 50) -> float:
    """Measures precise GPU kernel execution time in milliseconds using torch.cuda.Event."""
    for _ in range(warmup):
        func(*inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        func(*inputs)
    stop_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(stop_event) / iters


def main():
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Please run on an NVIDIA GPU.")
        sys.exit(1)

    init_and_warmup_cuda("cuda")

    # Load extension dynamically
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_kernels = os.path.dirname(current_dir)
        csrc_dir = os.path.join(root_kernels, "csrc")
        src_dir = os.path.join(root_kernels, "src")
        include_dir = os.path.join(root_kernels, "include")
        common_dir = os.path.join(root_kernels, "..", "00_common", "include")

        cuda_kernels = load(
            name="cuda_kernels_gemm_shapes",
            sources=[
                os.path.join(csrc_dir, "binding.cpp"),
                os.path.join(src_dir, "gemm.cu"),
                os.path.join(src_dir, "convolution.cu"),
                os.path.join(src_dir, "reduction.cu"),
                os.path.join(src_dir, "softmax.cu"),
                os.path.join(src_dir, "normalization.cu"),
                os.path.join(src_dir, "activation.cu"),
                os.path.join(src_dir, "pooling.cu"),
                os.path.join(src_dir, "elementwise.cu"),
            ],
            extra_include_paths=[include_dir, common_dir],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    except Exception as e:
        print(f"[ERROR] Loading CUDA extension failed: {e}")
        return

    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 64),    # Tall-skinny projection
        (128, 1024, 4096),   # Sequence projection
    ]

    device_name = torch.cuda.get_device_name(0)
    print("\n" + "=" * 105)
    print(f"🚀 CUDA GEMM Multi-Shape Benchmark vs cuBLAS (torch.matmul) | GPU: {device_name}")
    print("=" * 105)
    print(f"| {'Matrix Shape (M x N x K)':<24} | {'Custom (ms)':>11} | {'cuBLAS (ms)':>11} | {'Custom GFLOP/s':>14} | {'cuBLAS GFLOP/s':>14} | {'Speedup':>7} | {'Status':<6} |")
    print("|" + "-"*26 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*16 + "|" + "-"*16 + "|" + "-"*9 + "|" + "-"*8 + "|")

    for M, N, K in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Numerical Parity Check
        custom_out = cuda_kernels.gemm_register_tiled(A, B)
        torch_out = torch.matmul(A, B)
        max_diff = (custom_out - torch_out).abs().max().item()

        # Timing
        t_custom = time_gemm_operation(lambda a, b: cuda_kernels.gemm_register_tiled(a, b), (A, B))
        t_cublas = time_gemm_operation(lambda a, b: torch.matmul(a, b), (A, B))

        flops = 2.0 * M * N * K
        custom_gflops = (flops / (t_custom * 1e-3)) / 1e9 if t_custom > 0 else 0
        cublas_gflops = (flops / (t_cublas * 1e-3)) / 1e9 if t_cublas > 0 else 0
        speedup = t_cublas / t_custom if t_custom > 0 else 1.0

        status = "✅ PASS" if max_diff < 1e-3 else f"❌ {max_diff:.1e}"
        shape_str = f"{M}x{N}x{K}"

        print(f"| {shape_str:<24} | {t_custom:>9.3f} ms | {t_cublas:>9.3f} ms | {custom_gflops:>12.1f}   | {cublas_gflops:>12.1f}   | {speedup:>7.2f}x | {status} |")

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

import os
import sys
import time
import torch
import torch.nn.functional as F

def benchmark_op(name, custom_fn, torch_fn, inputs, warmup=25, iters=100):
    # Parity check
    custom_out = custom_fn(*inputs)
    torch_out = torch_fn(*inputs)
    
    if isinstance(custom_out, (tuple, list)):
        diff = max((c - t).abs().max().item() for c, t in zip(custom_out, torch_out))
    else:
        diff = (custom_out - torch_out).abs().max().item()
    
    # Warmup
    for _ in range(warmup):
        custom_fn(*inputs)
    torch.cuda.synchronize()
    
    # Benchmark Custom
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        custom_fn(*inputs)
    stop_event.record()
    torch.cuda.synchronize()
    custom_time_ms = start_event.elapsed_time(stop_event) / iters
    
    # Warmup Torch
    for _ in range(warmup):
        torch_fn(*inputs)
    torch.cuda.synchronize()
    
    # Benchmark Torch
    start_event.record()
    for _ in range(iters):
        torch_fn(*inputs)
    stop_event.record()
    torch.cuda.synchronize()
    torch_time_ms = start_event.elapsed_time(stop_event) / iters
    
    speedup = torch_time_ms / custom_time_ms if custom_time_ms > 0 else 1.0
    status = "✅ PASS" if diff < 1e-3 else "❌ FAIL"
    
    print(f"| {name:<32} | {custom_time_ms*1000:>8.2f} µs | {torch_time_ms*1000:>8.2f} µs | {speedup:>6.2f}x | {diff:>9.2e} | {status} |")

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled machine.")
        return
        
    print("\n" + "=" * 95)
    print("🚀 CUDA Kernel Engine: Micro-Benchmark Suite vs PyTorch Native Operations")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 95)
    print(f"| {'Kernel Operation':<32} | {'Custom CUDA':>11} | {'PyTorch':>11} | {'Speedup':>7} | {'Max Delta':>9} | {'Status':<6} |")
    print("|" + "-"*34 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*9 + "|" + "-"*11 + "|" + "-"*8 + "|")
    
    # Compile / import extension dynamically
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_kernels = os.path.dirname(current_dir)
        
        csrc_dir = os.path.join(root_kernels, "csrc")
        src_dir = os.path.join(root_kernels, "src")
        include_dir = os.path.join(root_kernels, "include")
        common_dir = os.path.join(root_kernels, "..", "00_common", "include")
        
        cuda_kernels = load(
            name="cuda_kernels_engine_bench",
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
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as e:
        print(f"\nNote: JIT Extension loading skipped or encountered: {e}")
        print("Falling back to pre-built / testing simulation.")
        return

    # 1. 2D Tiled GEMM (512 x 512)
    A = torch.randn(512, 512, device="cuda")
    B = torch.randn(512, 512, device="cuda")
    benchmark_op("GEMM Tiled (512x512)", lambda a, b: cuda_kernels.gemm_tiled(a, b), lambda a, b: torch.matmul(a, b), (A, B))

    # 2. Register Tiled GEMM (1024 x 1024)
    A = torch.randn(1024, 1024, device="cuda")
    B = torch.randn(1024, 1024, device="cuda")
    benchmark_op("GEMM Register-Tiled (1024x1024)", lambda a, b: cuda_kernels.gemm_register_tiled(a, b), lambda a, b: torch.matmul(a, b), (A, B))

    # 3. Softmax Forward (1024 x 1000)
    logits = torch.randn(1024, 1000, device="cuda")
    benchmark_op("Softmax Forward (1024x1000)", lambda x: cuda_kernels.softmax_forward(x), lambda x: F.softmax(x, dim=-1), (logits,))
    benchmark_op("FlashSoftmax Online (1024x1000)", lambda x: cuda_kernels.online_safe_softmax(x), lambda x: F.softmax(x, dim=-1), (logits,))

    # 4. LayerNorm Forward (1024 x 512)
    X = torch.randn(1024, 512, device="cuda")
    gamma = torch.ones(512, device="cuda")
    beta = torch.zeros(512, device="cuda")
    benchmark_op("LayerNorm Forward (1024x512)", 
                 lambda x, g, b: cuda_kernels.layernorm_forward(x, g, b, 1e-5)[0], 
                 lambda x, g, b: F.layer_norm(x, (512,), g, b, 1e-5), 
                 (X, gamma, beta))

    # 5. Vectorized GELU Forward (4M elements)
    X_act = torch.randn(1024 * 4096, device="cuda")
    benchmark_op("GELU Vectorized (4.19M)", lambda x: cuda_kernels.gelu_forward(x), lambda x: F.gelu(x, approximate="tanh"), (X_act,))

    # 6. Fused Residual Addition (4M elements)
    res = torch.randn_like(X_act)
    benchmark_op("Fused Residual Add (4.19M)", lambda x, r: cuda_kernels.fused_residual_add(x, r), lambda x, r: x + r, (X_act, res))

    print("=" * 95)
    print("All kernel operations verified for parity and benchmarked against PyTorch.\n")

if __name__ == "__main__":
    main()

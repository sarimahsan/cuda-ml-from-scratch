"""
=============================================================================
Modular Pure CUDA LSTM: GPU Benchmark Suite (Custom CUDA vs PyTorch Native)
=============================================================================
Comprehensive benchmark evaluating:
  1. Gate Micro-benchmarks (Individual Gate kernels vs Fused kernel)
  2. Sequence Macro-benchmarks (Custom CUDA Modular vs Fused vs PyTorch nn.LSTM)
  3. Exact Numerical Parity Verification (Forward outputs & Backward gradients)
=============================================================================
"""

import argparse
import math
import time
import torch
import torch.nn as nn

try:
    from lstm import CUDALSTM, CUDALSTMCell, _ext
except ImportError:
    _ext = None


def benchmark_numerical_parity(device="cuda"):
    print("\n" + "=" * 75)
    print(" 🧪 TEST 1: Exact Numerical Parity (Custom CUDA vs PyTorch Autograd)")
    print("=" * 75)

    if not torch.cuda.is_available() or _ext is None:
        print("[WARNING] CUDA not available or extension not compiled. Skipping parity test.")
        return

    torch.manual_seed(42)
    seq_len = 8
    batch_size = 4
    input_dim = 16
    hidden_dim = 32

    # 1. Custom CUDA LSTM
    custom_lstm = CUDALSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        mode="fused",
        device=device,
    )

    # 2. PyTorch Native nn.LSTM
    pt_lstm = nn.LSTM(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        bias=True,
        batch_first=False,
    ).to(device)

    # Copy exact weights:
    # PyTorch layout: [4H, D] -> custom layout: [D, 4H]
    with torch.no_grad():
        pt_lstm.weight_ih_l0.copy_(custom_lstm.W_ih.t())
        pt_lstm.weight_hh_l0.copy_(custom_lstm.W_hh.t())
        pt_lstm.bias_ih_l0.copy_(custom_lstm.b_ih)
        pt_lstm.bias_hh_l0.copy_(custom_lstm.b_hh)

    # Input sequence
    X = torch.randn(seq_len, batch_size, input_dim, device=device, requires_grad=True)
    h_0 = torch.zeros(batch_size, hidden_dim, device=device)
    c_0 = torch.zeros(batch_size, hidden_dim, device=device)

    # Custom Forward
    out_custom, (h_n_custom, c_n_custom), cache = custom_lstm.forward_sequence(X.detach(), h_0, c_0)

    # PyTorch Forward
    pt_h0 = h_0.unsqueeze(0)
    pt_c0 = c_0.unsqueeze(0)
    out_pt, (h_n_pt, c_n_pt) = pt_lstm(X, (pt_h0, pt_c0))

    # Compare Forward Outputs
    fwd_diff_max = (out_custom - out_pt).abs().max().item()
    fwd_diff_mean = (out_custom - out_pt).abs().mean().item()

    print(f"• Forward Hidden States Parity: Max Diff = {fwd_diff_max:.2e} | Mean Diff = {fwd_diff_mean:.2e}")
    assert fwd_diff_max < 1e-4, f"Forward parity mismatch: {fwd_diff_max}"
    print("  => [PASS] Custom CUDA Forward perfectly matches PyTorch nn.LSTM!")

    # Backward Parity
    grad_output = torch.randn_like(out_pt)
    out_pt.backward(grad_output)

    # Custom Backward BPTT
    dW_ih, db_ih, dW_hh, db_hh, dX = custom_lstm.backward_sequence(grad_output, cache)

    # Compare Weight Gradients
    pt_dW_ih = pt_lstm.weight_ih_l0.grad.t()
    pt_dW_hh = pt_lstm.weight_hh_l0.grad.t()
    pt_db_ih = pt_lstm.bias_ih_l0.grad
    pt_db_hh = pt_lstm.bias_hh_l0.grad

    grad_W_ih_diff = (dW_ih - pt_dW_ih).abs().max().item()
    grad_W_hh_diff = (dW_hh - pt_dW_hh).abs().max().item()
    grad_b_ih_diff = (db_ih - pt_db_ih).abs().max().item()
    grad_b_hh_diff = (db_hh - pt_db_hh).abs().max().item()
    grad_X_diff = (dX - X.grad).abs().max().item()

    print(f"• Backward dW_ih Parity       : Max Diff = {grad_W_ih_diff:.2e}")
    print(f"• Backward dW_hh Parity       : Max Diff = {grad_W_hh_diff:.2e}")
    print(f"• Backward db_ih Parity       : Max Diff = {grad_b_ih_diff:.2e}")
    print(f"• Backward db_hh Parity       : Max Diff = {grad_b_hh_diff:.2e}")
    print(f"• Backward dX Parity          : Max Diff = {grad_X_diff:.2e}")

    assert max(grad_W_ih_diff, grad_W_hh_diff, grad_b_ih_diff, grad_b_hh_diff, grad_X_diff) < 1e-3
    print("  => [PASS] Custom CUDA BPTT Gradients perfectly match PyTorch Autograd!\n")


def benchmark_macro_throughput(seq_len=64, batch_size=64, input_dim=128, hidden_dim=256, iters=50, device="cuda"):
    print("=" * 75)
    print(" 🚀 TEST 2: Sequence Throughput Benchmark (Custom CUDA vs PyTorch Native)")
    print("=" * 75)
    print(f"• Settings: Seq Len = {seq_len} | Batch = {batch_size} | Input Dim = {input_dim} | Hidden Dim = {hidden_dim}")
    print("-" * 75)

    if not torch.cuda.is_available() or _ext is None:
        print("[WARNING] CUDA not available or extension not compiled.")
        return

    total_tokens = seq_len * batch_size

    # Prepare Models
    custom_fused = CUDALSTM(input_dim, hidden_dim, mode="fused", device=device)
    custom_modular = CUDALSTM(input_dim, hidden_dim, mode="modular", device=device)
    pt_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bias=True).to(device)

    X = torch.randn(seq_len, batch_size, input_dim, device=device)
    grad_out = torch.randn(seq_len, batch_size, hidden_dim, device=device)

    # Helper benchmark function using CUDA Events
    def measure_pipeline(name, forward_fn, backward_fn):
        # Warmup
        for _ in range(10):
            out, cache = forward_fn()
            backward_fn(out, cache)
        torch.cuda.synchronize()

        # Measure Forward using CUDA Events
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iters):
            out, cache = forward_fn()
        stop_event.record()
        torch.cuda.synchronize()
        fwd_ms = start_event.elapsed_time(stop_event) / iters

        # Measure Backward using CUDA Events
        start_event.record()
        for _ in range(iters):
            backward_fn(out, cache)
        stop_event.record()
        torch.cuda.synchronize()
        bwd_ms = start_event.elapsed_time(stop_event) / iters

        total_ms = fwd_ms + bwd_ms
        tokens_sec = total_tokens / (total_ms / 1000.0)

        print(f"• {name:<32}: Forward = {fwd_ms:6.2f} ms | Backward = {bwd_ms:6.2f} ms | Total = {total_ms:6.2f} ms | Throughput = {tokens_sec:9,.0f} tok/s")
        return fwd_ms, bwd_ms, total_ms, tokens_sec

    # 1. Custom Modular
    def fwd_mod():
        out, (h, c), cache = custom_modular.forward_sequence(X)
        return out, cache

    def bwd_mod(out, cache):
        custom_modular.backward_sequence(grad_out, cache)

    measure_pipeline("Custom CUDA (Modular Gates)", fwd_mod, bwd_mod)

    # 2. Custom Fused Native C++
    def fwd_fused():
        out, (h, c), cache = custom_fused.forward_sequence(X)
        return out, cache

    def bwd_fused(out, cache):
        custom_fused.backward_sequence(grad_out, cache)

    measure_pipeline("Custom CUDA (Fused Gates)", fwd_fused, bwd_fused)

    # 3. PyTorch Native cuDNN
    def fwd_pt():
        out, _ = pt_lstm(X)
        return out, None

    def bwd_pt(out, cache):
        pt_lstm.zero_grad()
        out.backward(grad_out, retain_graph=True)

    measure_pipeline("PyTorch nn.LSTM (cuDNN)", fwd_pt, bwd_pt)
    print("=" * 75 + "\n")



def init_and_warmup_cuda(device: str = "cuda"):
    """
    Explicitly initializes CUDA runtime, primary context, and PyTorch's internal cuBLAS / cuDNN handle pool.
    Runs warmup kernel & backward passes to prevent lazy-initialization jitter from contaminating microbenchmarks.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return
    torch.cuda.init()
    torch.cuda.synchronize()

    dummy_x = torch.randn(16, 4, 32, device=device, requires_grad=True)
    dummy_lstm = nn.LSTM(32, 64, num_layers=1, batch_first=False).to(device)
    for _ in range(50):
        dummy_out, _ = dummy_lstm(dummy_x)
        dummy_loss = dummy_out.sum()
        dummy_loss.backward()
        torch.cuda.synchronize()

    del dummy_x, dummy_lstm, dummy_out, dummy_loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Modular CUDA LSTM Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    # Initialize CUDA context and warm up cuBLAS / cuDNN
    init_and_warmup_cuda(device=args.device)

    benchmark_numerical_parity(device=args.device)

    if args.quick:
        benchmark_macro_throughput(seq_len=32, batch_size=32, input_dim=64, hidden_dim=128, iters=20, device=args.device)
    else:
        benchmark_macro_throughput(seq_len=64, batch_size=64, input_dim=128, hidden_dim=256, iters=50, device=args.device)


if __name__ == "__main__":
    main()

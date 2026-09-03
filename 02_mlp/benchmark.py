"""
=============================================================================
CUDA ML vs PyTorch Native: Multi-Layer Perceptron (MLP) Comprehensive Benchmark
=============================================================================
This script benchmarks the custom modular CUDA C++ MLP kernels against
PyTorch's native cuBLAS-backed nn.Linear, Autograd, and PyTorch C10 engine.

Comparisons:
  1. Kernel-level Microbenchmarks:
     - Tiled Shared-Memory GEMM (Forward & Backward)
     - Elementwise Activations (ReLU, GELU, Sigmoid Forward & Backward)
     - Numerically Stable Fused Softmax Cross-Entropy & Gradients
     - In-place Vectorized Optimizers (Adam, SGD with Momentum)
  2. Batch & Architecture Scaling Macrobenchmarks:
     - Small, Standard, and Deep Architectures across batch sizes (64 to 2048)
  3. End-to-End MNIST Training Benchmark:
     - Real / Synthetic MNIST training convergence, throughput, and accuracy
  4. Peak VRAM Memory Profiling
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

# Import custom CUDA MLP extension & class
try:
    from mlp import CUDAMLP, _ext
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from mlp import CUDAMLP, _ext


# ---------------------------------------------------------------------------
# Native PyTorch MLP Baseline Module
# ---------------------------------------------------------------------------
class PyTorchMLP(nn.Module):
    """
    Standard PyTorch baseline matching the exact layer architectures of CUDAMLP.
    """

    def __init__(self, layer_sizes: List[int], activation: str = "relu", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        layers: List[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True, device=self.device))
            if i < len(layer_sizes) - 2:
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "gelu":
                    layers.append(nn.GELU())
                elif activation.lower() == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# High-Precision CUDA Timing Utility
# ---------------------------------------------------------------------------
def time_cuda_operation(op_func, warmup: int = 10, reps: int = 50) -> Tuple[float, float]:
    """
    Measures GPU execution time using torch.cuda.Event with warmup and synchronization.
    Returns:
        (mean_ms, std_ms)
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
# Benchmark 1: Kernel-Level Microbenchmarks
# ---------------------------------------------------------------------------
def benchmark_kernels(M: int = 512, K: int = 784, N: int = 256, device: str = "cuda") -> List[Dict]:
    """
    Microbenchmark individual CUDA kernels against PyTorch native equivalents:
      - Linear GEMM Forward & Backward
      - Activations (ReLU, GELU, Sigmoid)
      - Fused Softmax Cross-Entropy Loss + Gradient
      - Optimizers (Adam, SGD Momentum)
    """
    print("=" * 85)
    print(f" 1. KERNEL-LEVEL MICROBENCHMARKS: Custom CUDA C++ vs PyTorch Native")
    print(f"    Dimensions: Batch M={M}, Fan-in K={K}, Fan-out N={N}")
    print("=" * 85)

    dev = torch.device(device)
    torch.manual_seed(42)

    X = torch.randn(M, K, device=dev, dtype=torch.float32)
    W = torch.randn(K, N, device=dev, dtype=torch.float32)
    b = torch.randn(N, device=dev, dtype=torch.float32)
    dZ = torch.randn(M, N, device=dev, dtype=torch.float32)
    targets = torch.randint(0, N, (M,), device=dev, dtype=torch.int64)

    # PyTorch reference layer
    pt_linear = nn.Linear(K, N, bias=True, device=dev)
    with torch.no_grad():
        pt_linear.weight.copy_(W.t())
        pt_linear.bias.copy_(b)

    results = []

    # 1. Tiled GEMM Forward: Z = X * W + b
    def cuda_gemm_fwd():
        return _ext.linear_forward(X, W, b)

    def pt_gemm_fwd():
        return pt_linear(X)

    t_c, _ = time_cuda_operation(cuda_gemm_fwd)
    t_p, _ = time_cuda_operation(pt_gemm_fwd)
    results.append({"Kernel": "Linear Forward GEMM (Z = XW + b)", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # 2. Linear Backward: dW, db, dX
    def cuda_gemm_bwd():
        return _ext.linear_backward(dZ, X, W, True)

    out_pt = pt_linear(X)

    def pt_gemm_bwd():
        pt_linear.zero_grad(set_to_none=False)
        out_pt.backward(dZ, retain_graph=True)

    t_c, _ = time_cuda_operation(cuda_gemm_bwd)
    t_p, _ = time_cuda_operation(pt_gemm_bwd)
    results.append({"Kernel": "Linear Backward GEMM (dW, db, dX)", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # 3. Activations (ReLU, GELU, Sigmoid)
    Z = cuda_gemm_fwd()
    dA = torch.randn_like(Z)

    # ReLU Forward
    t_c, _ = time_cuda_operation(lambda: _ext.relu_forward(Z))
    t_p, _ = time_cuda_operation(lambda: torch.relu(Z))
    results.append({"Kernel": "Activation: ReLU Forward", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # ReLU Backward
    t_c, _ = time_cuda_operation(lambda: _ext.relu_backward(dA, Z))
    relu_z = Z.clone().requires_grad_(True)
    out_relu = torch.relu(relu_z)
    t_p, _ = time_cuda_operation(lambda: out_relu.backward(dA, retain_graph=True))
    results.append({"Kernel": "Activation: ReLU Backward", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # GELU Forward
    t_c, _ = time_cuda_operation(lambda: _ext.gelu_forward(Z))
    t_p, _ = time_cuda_operation(lambda: torch.nn.functional.gelu(Z))
    results.append({"Kernel": "Activation: GELU Forward", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # 4. Softmax + Cross-Entropy Loss & Fused Gradients
    ce_loss_fn = nn.CrossEntropyLoss()
    logits = torch.randn(M, N, device=dev, dtype=torch.float32)

    def cuda_softmax_ce():
        return _ext.softmax_cross_entropy(logits, targets)

    def pt_softmax_ce():
        logits_pt = logits.clone().requires_grad_(True)
        l = ce_loss_fn(logits_pt, targets)
        l.backward()
        return l

    t_c, _ = time_cuda_operation(cuda_softmax_ce)
    t_p, _ = time_cuda_operation(pt_softmax_ce)
    results.append({"Kernel": "Fused Softmax + CE Loss & Gradients", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # 5. Optimizers (Adam and SGD with Momentum)
    param_cuda = W.clone()
    m_cuda = torch.zeros_like(W)
    v_cuda = torch.zeros_like(W)
    grad_cuda = torch.randn_like(W)

    def cuda_adam():
        _ext.adam_step(param_cuda, m_cuda, v_cuda, grad_cuda, 1e-3, 0.9, 0.999, 1e-8, 1)

    pt_param = nn.Parameter(W.clone())
    pt_adam = optim.Adam([pt_param], lr=1e-3)
    pt_param.grad = grad_cuda.clone()

    def pt_adam_step():
        pt_adam.step()

    t_c, _ = time_cuda_operation(cuda_adam)
    t_p, _ = time_cuda_operation(pt_adam_step)
    results.append({"Kernel": "Optimizer: Adam In-Place Step", "CUDA (ms)": t_c, "PyTorch (ms)": t_p, "Speedup": t_p / t_c if t_c > 0 else 0})

    # Print Microbenchmark Table
    print(f"{'Kernel Operation':<42} | {'CUDA (ms)':<10} | {'PyTorch (ms)':<12} | {'Speedup':<8}")
    print("-" * 85)
    for r in results:
        print(f"{r['Kernel']:<42} | {r['CUDA (ms)']:>10.4f} | {r['PyTorch (ms)']:>12.4f} | {r['Speedup']:>7.2f}x")
    print("-" * 85)

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Architecture & Batch Size Scaling
# ---------------------------------------------------------------------------
def benchmark_architecture_scaling(device: str = "cuda") -> List[Dict]:
    """
    Benchmarks full forward+backward+optimizer training step across:
      - 3 Architectures: Small [784, 128, 10], Medium [784, 256, 128, 10], Deep [1024, 1024, 512, 256, 10]
      - 4 Batch Sizes: 64, 128, 512, 2048
    """
    print("\n" + "=" * 90)
    print(" 2. MACROBENCHMARK: Architecture & Batch Size Throughput Scaling")
    print("=" * 90)

    dev = torch.device(device)
    architectures = [
        ("Small MLP", [784, 128, 10]),
        ("Standard MNIST MLP", [784, 256, 128, 10]),
        ("Deep/Wide MLP", [1024, 1024, 512, 256, 10]),
    ]
    batch_sizes = [64, 128, 512, 2048]

    all_results = []

    for arch_name, layer_sizes in architectures:
        print(f"\n--- Model Architecture: {arch_name} ({' -> '.join(map(str, layer_sizes))}) ---")
        print(f"{'Batch Size':<12} | {'CUDA (ms)':<10} | {'PyTorch (ms)':<12} | {'CUDA Thrpt (img/s)':<19} | {'Speedup':<8}")
        print("-" * 80)

        for B in batch_sizes:
            in_dim = layer_sizes[0]
            out_dim = layer_sizes[-1]

            X = torch.randn(B, in_dim, device=dev, dtype=torch.float32)
            y = torch.randint(0, out_dim, (B,), device=dev, dtype=torch.int64)

            # Custom CUDA MLP
            cuda_mlp = CUDAMLP(layer_sizes=layer_sizes, activation="relu", lr=1e-3, optimizer="adam", device=device)

            # PyTorch MLP
            pt_mlp = PyTorchMLP(layer_sizes=layer_sizes, activation="relu", device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(pt_mlp.parameters(), lr=1e-3)

            # 1. CUDA Full Step
            def run_cuda_step():
                # Forward
                acts, pre_acts = cuda_mlp.forward_pass(X)
                logits = acts[-1]
                # Softmax + Loss
                probs, loss, dZ = _ext.softmax_cross_entropy(logits, y)
                # Backward
                cur_dZ = dZ
                grads_W = [None] * cuda_mlp.num_layers
                grads_b = [None] * cuda_mlp.num_layers
                for layer_idx in reversed(range(cuda_mlp.num_layers)):
                    A_in = acts[layer_idx]
                    W = cuda_mlp.weights[layer_idx]
                    comp_dX = layer_idx > 0
                    dW, db, dX = _ext.linear_backward(cur_dZ, A_in, W, comp_dX)
                    grads_W[layer_idx] = dW
                    grads_b[layer_idx] = db
                    if comp_dX:
                        Z_prev = pre_acts[layer_idx - 1]
                        cur_dZ = _ext.relu_backward(dX, Z_prev)
                # Optimizer Step
                cuda_mlp.step_count += 1
                for layer_idx in range(cuda_mlp.num_layers):
                    _ext.adam_step(
                        cuda_mlp.weights[layer_idx],
                        cuda_mlp.m_w[layer_idx],
                        cuda_mlp.v_w[layer_idx],
                        grads_W[layer_idx],
                        cuda_mlp.lr,
                        0.9,
                        0.999,
                        1e-8,
                        cuda_mlp.step_count,
                    )
                    _ext.adam_step(
                        cuda_mlp.biases[layer_idx],
                        cuda_mlp.m_b[layer_idx],
                        cuda_mlp.v_b[layer_idx],
                        grads_b[layer_idx],
                        cuda_mlp.lr,
                        0.9,
                        0.999,
                        1e-8,
                        cuda_mlp.step_count,
                    )

            # 2. PyTorch Full Step
            def run_pt_step():
                optimizer.zero_grad(set_to_none=True)
                logits = pt_mlp(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            t_cuda, _ = time_cuda_operation(run_cuda_step, reps=30)
            t_pt, _ = time_cuda_operation(run_pt_step, reps=30)
            speedup = t_pt / t_cuda if t_cuda > 0 else 0.0
            throughput = B / (t_cuda / 1000.0)

            all_results.append({
                "Architecture": arch_name,
                "Batch Size": B,
                "CUDA (ms)": t_cuda,
                "PyTorch (ms)": t_pt,
                "Throughput (img/s)": throughput,
                "Speedup": speedup,
            })

            print(f"{B:<12d} | {t_cuda:>10.4f} | {t_pt:>12.4f} | {throughput:>19.1f} | {speedup:>7.2f}x")

    return all_results


# ---------------------------------------------------------------------------
# Benchmark 3: End-to-End MNIST Training Benchmark & Convergence
# ---------------------------------------------------------------------------
def benchmark_mnist_training(epochs: int = 5, batch_size: int = 128, device: str = "cuda"):
    """
    End-to-End training on MNIST (or synthetic digits) comparing:
      - Epoch-by-epoch loss convergence
      - Final test accuracy
      - Total training wall-clock time
    """
    print("\n" + "=" * 85)
    print(f" 3. END-TO-END MNIST TRAINING BENCHMARK ({epochs} Epochs, Batch Size={batch_size})")
    print("=" * 85)

    # 1. Dataset generation / loading
    print("[INFO] Preparing MNIST dataset...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        X_train = train_dataset.data.float().view(-1, 28 * 28) / 255.0
        y_train = train_dataset.targets.long()
        X_test = test_dataset.data.float().view(-1, 28 * 28) / 255.0
        y_test = test_dataset.targets.long()
    except Exception:
        print("[WARNING] Falling back to synthetic 784-dim digits dataset (N=60,000 train, 10,000 test).")
        np.random.seed(42)
        N_train, N_test, D, C = 60000, 10000, 784, 10
        centers = np.random.randn(C, D).astype(np.float32)
        y_tr = np.random.randint(0, C, size=N_train)
        X_tr = centers[y_tr] + np.random.randn(N_train, D).astype(np.float32) * 0.5
        y_te = np.random.randint(0, C, size=N_test)
        X_te = centers[y_te] + np.random.randn(N_test, D).astype(np.float32) * 0.5
        X_train, y_train = torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()
        X_test, y_test = torch.from_numpy(X_te), torch.from_numpy(y_te).long()

    layer_sizes = [784, 256, 128, 10]
    lr = 1e-3

    # --- Train Custom CUDA MLP ---
    print("\n--- Training Model A: Custom Modular CUDA MLP ---")
    cuda_model = CUDAMLP(layer_sizes=layer_sizes, lr=lr, optimizer="adam", device=device)

    t0 = time.perf_counter()
    cuda_model.fit(X=X_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=True)
    cuda_total_time = time.perf_counter() - t0
    cuda_test_acc = cuda_model.score(X_test, y_test)

    # --- Train Native PyTorch MLP ---
    print("\n--- Training Model B: Native PyTorch Autograd MLP ---")
    pt_model = PyTorchMLP(layer_sizes=layer_sizes, activation="relu", device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pt_model.parameters(), lr=lr)

    X_tr_dev = X_train.to(device)
    y_tr_dev = y_train.to(device)
    X_te_dev = X_test.to(device)
    y_te_dev = y_test.to(device)

    N = X_train.size(0)
    num_batches = (N + batch_size - 1) // batch_size

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=torch.device(device))
        X_shuff = X_tr_dev[perm]
        y_shuff = y_tr_dev[perm]
        epoch_loss = 0.0

        for b in range(num_batches):
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, N)
            bx = X_shuff[s_idx:e_idx]
            by = y_shuff[s_idx:e_idx]

            optimizer.zero_grad(set_to_none=True)
            logits = pt_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * (e_idx - s_idx)

        # Validation
        with torch.no_grad():
            preds = pt_model(X_te_dev).argmax(dim=1)
            val_acc = (preds == y_te_dev).float().mean().item() * 100.0

        print(f"  Epoch {epoch:3d}/{epochs:3d} | CE Loss: {epoch_loss / N:.4f} | Val Acc: {val_acc:6.2f}%")

    pt_total_time = time.perf_counter() - t0

    with torch.no_grad():
        pt_preds = pt_model(X_te_dev).argmax(dim=1)
        pt_test_acc = (pt_preds == y_te_dev).float().mean().item() * 100.0

    print("\n" + "=" * 80)
    print(" END-TO-END TRAINING COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Custom CUDA MLP':<20} | {'PyTorch Native MLP':<20}")
    print("-" * 80)
    print(f"{'Total Training Time (s)':<30} | {cuda_total_time:>20.3f} | {pt_total_time:>20.3f}")
    print(f"{'Time per Epoch (ms)':<30} | {(cuda_total_time/epochs)*1000:>20.1f} | {(pt_total_time/epochs)*1000:>20.1f}")
    print(f"{'Final Test Accuracy (%)':<30} | {cuda_test_acc:>19.2f}% | {pt_test_acc:>19.2f}%")
    speedup = pt_total_time / cuda_total_time if cuda_total_time > 0 else 0.0
    print(f"{'Relative Speedup Factor':<30} | {speedup:>19.2f}x | {'1.00x (Baseline)':>20}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Benchmark 4: Peak GPU VRAM Profiling
# ---------------------------------------------------------------------------
def benchmark_memory(B: int = 1024, layer_sizes: List[int] = [1024, 1024, 512, 10], device: str = "cuda"):
    """
    Measures and compares peak GPU memory allocated during a forward+backward step.
    """
    print("\n" + "=" * 80)
    print(f" 4. PEAK GPU MEMORY CONSUMPTION PROFILE (Batch={B}, Architecture={layer_sizes})")
    print("=" * 80)

    dev = torch.device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    # 1. Custom CUDA MLP Memory
    cuda_mlp = CUDAMLP(layer_sizes=layer_sizes, activation="relu", optimizer="adam", device=device)
    X = torch.randn(B, layer_sizes[0], device=dev, dtype=torch.float32)
    y = torch.randint(0, layer_sizes[-1], (B,), device=dev, dtype=torch.int64)

    torch.cuda.reset_peak_memory_stats(dev)
    acts, pre_acts = cuda_mlp.forward_pass(X)
    probs, loss, dZ = _ext.softmax_cross_entropy(acts[-1], y)
    cur_dZ = dZ
    for layer_idx in reversed(range(cuda_mlp.num_layers)):
        A_in = acts[layer_idx]
        W = cuda_mlp.weights[layer_idx]
        comp_dX = layer_idx > 0
        dW, db, dX = _ext.linear_backward(cur_dZ, A_in, W, comp_dX)
        if comp_dX:
            cur_dZ = _ext.relu_backward(dX, pre_acts[layer_idx - 1])
    torch.cuda.synchronize()
    cuda_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    # Clean up
    del cuda_mlp, X, y, acts, pre_acts, probs, loss, dZ, cur_dZ
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    # 2. PyTorch Native MLP Memory
    pt_mlp = PyTorchMLP(layer_sizes=layer_sizes, activation="relu", device=device)
    criterion = nn.CrossEntropyLoss()
    X_pt = torch.randn(B, layer_sizes[0], device=dev, dtype=torch.float32)
    y_pt = torch.randint(0, layer_sizes[-1], (B,), device=dev, dtype=torch.int64)

    torch.cuda.reset_peak_memory_stats(dev)
    out = pt_mlp(X_pt)
    loss_pt = criterion(out, y_pt)
    loss_pt.backward()
    torch.cuda.synchronize()
    pt_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    print(f"• Custom CUDA Kernel Peak VRAM Allocation   : {cuda_peak_mb:.2f} MB")
    print(f"• PyTorch Native Autograd Peak VRAM         : {pt_peak_mb:.2f} MB")
    print(f"• Memory Savings with Custom In-Place Kernels: {pt_peak_mb - cuda_peak_mb:+.2f} MB")
    print("=" * 80)


# ---------------------------------------------------------------------------
def init_and_warmup_cuda(device: str = "cuda"):
    """
    Explicitly initializes CUDA runtime, primary context, and PyTorch's internal cuBLAS / cuDNN handle pool.
    Runs warmup kernel & backward passes to prevent lazy-initialization jitter from contaminating microbenchmarks.
    """
    if not torch.cuda.is_available() or device == "cpu":
        return
    torch.cuda.init()
    torch.cuda.synchronize()

    # Warm up CUDA driver, memory allocator, cuBLAS GEMM, and Autograd engines
    dummy_a = torch.randn(256, 256, device=device, requires_grad=True)
    dummy_b = torch.randn(256, 256, device=device, requires_grad=True)
    for _ in range(50):
        dummy_c = torch.matmul(dummy_a, dummy_b)
        dummy_loss = dummy_c.sum()
        dummy_loss.backward()
        torch.cuda.synchronize()

    del dummy_a, dummy_b, dummy_c, dummy_loss
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CUDA ML vs PyTorch Multi-Layer Perceptron (MLP) Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--quick", action="store_true", help="Run a quick version with fewer iterations")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for MNIST training benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. Please run on a GPU-enabled runtime (e.g. Google Colab).")
        sys.exit(1)

    # Initialize CUDA context and warm up cuBLAS before timing
    init_and_warmup_cuda(device=args.device)

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 85)
    print("      🚀 CUDA ML BENCHMARK SUITE: MULTI-LAYER PERCEPTRON (MLP) 🚀")
    print("=" * 85)
    print(f"• GPU Device         : {device_name}")
    print(f"• Compute Capability : {device_props.major}.{device_props.minor}")
    print(f"• Total VRAM         : {device_props.total_memory / (1024**3):.2f} GB")
    print(f"• PyTorch Version    : {torch.__version__}")
    print(f"• CUDA Version       : {torch.version.cuda}")
    print("=" * 85 + "\n")

    # 1. Kernel-level microbenchmarks
    benchmark_kernels(M=256 if args.quick else 512, K=784, N=256, device=args.device)

    # 2. Architecture & Batch Size Scaling
    benchmark_architecture_scaling(device=args.device)

    # 3. End-to-End MNIST Training
    benchmark_mnist_training(epochs=3 if args.quick else args.epochs, batch_size=128, device=args.device)

    # 4. Memory Profiling
    benchmark_memory(B=512 if args.quick else 1024, device=args.device)

    print("\n[SUCCESS] Multi-Layer Perceptron (MLP) Benchmarking Complete!\n")


if __name__ == "__main__":
    main()

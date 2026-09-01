# 💾 GPU Memory Hierarchy & Bandwidth Architecture: Pure CUDA LSTM

This document analyzes the GPU memory layouts, memory access coalescing, shared memory bank conflicts, and register caching strategies in our modular CUDA LSTM implementation.

---

## 1. 🗄️ Tensor Memory Layouts

All tensors are allocated as contiguous 32-bit floating-point arrays in row-major layout on device global memory:

| Tensor | Shape | Memory Footprint (FP32) | Layout Rationale |
| :--- | :--- | :--- | :--- |
| **Input Sequence $\mathbf{X}$** | $[T, N, D]$ | $4 \times T \times N \times D$ bytes | Timestep-first ordering enables contiguous slice $\mathbf{X}[t]$ per step |
| **Input Weights $\mathbf{W}_{ih}$** | $[D, 4H]$ | $4 \times D \times 4H$ bytes | Transposed layout enables direct GEMM $\mathbf{X}[t] \times \mathbf{W}_{ih}$ |
| **Recurrent Weights $\mathbf{W}_{hh}$** | $[H, 4H]$ | $4 \times H \times 4H$ bytes | Direct GEMM with $\mathbf{H}_{t-1} \in \mathbb{R}^{N \times H}$ |
| **Biases $\mathbf{b}_{ih}, \mathbf{b}_{hh}$** | $[4H]$ | $2 \times 4 \times 4H$ bytes | Broadcasted in register additions |
| **Gate Pre-activations $\mathbf{G}$** | $[T, N, 4H]$ | $4 \times T \times N \times 4H$ bytes | Consecutive blocks for $[i, f, g, o]$ gates |
| **Cell States $\mathbf{C}$** | $[T+1, N, H]$ | $4 \times (T+1) \times N \times H$ bytes | Stores initial state at index $0$ |
| **Hidden States $\mathbf{H}$** | $[T+1, N, H]$ | $4 \times (T+1) \times N \times H$ bytes | Stores initial state at index $0$ |

---

## 2. ⚡ Fused Gate Kernel vs. Unfused Modular Pipeline

In an unfused LSTM implementation, evaluating the 4 gates and cell state requires multiple separate kernel launches:
1. `launch_input_gate_forward` $\to$ Read $Z_i$, Write $i$
2. `launch_forget_gate_forward` $\to$ Read $Z_f$, Write $f$
3. `launch_candidate_gate_forward` $\to$ Read $Z_g$, Write $g$
4. `launch_output_gate_forward` $\to$ Read $Z_o$, Write $o$
5. `launch_cell_state_forward` $\to$ Read $f, c_{t-1}, i, g, o$, Write $c_t, \tanh(c_t), h_t$

### Memory Traffic Comparison per Step $t$:

$$\text{Unfused VRAM Operations} = 4 \times (N \times H \times 2) + (N \times H \times 8) = 16 \times N \times H \text{ elements}$$
$$\text{Fused VRAM Operations} = \underbrace{4 \times N \times H}_{\text{Read } \mathbf{G}_t} + \underbrace{N \times H}_{\text{Read } \mathbf{c}_{t-1}} + \underbrace{N \times H}_{\text{Write } \mathbf{c}_t} + \underbrace{N \times H}_{\text{Write } \mathbf{h}_t} = 7 \times N \times H \text{ elements}$$

> [!TIP]
> **56.25% Bandwidth Reduction**: Fusing all gate evaluations into `fused_lstm_gates_forward_kernel` reduces global memory bandwidth pressure by over $2.28\times$, keeping intermediate values in GPU register files.

---

## 3. 🧩 2D Tiled Shared-Memory Matrix Multiplication (GEMM)

For projections $\mathbf{X}_t \mathbf{W}_{ih}$ and $\mathbf{H}_{t-1} \mathbf{W}_{hh}$, we utilize a 2D tiled GEMM kernel with $16 \times 16$ tile dimensions:

```
                    Matrix B (Weights) [K x N]
                    ┌─────────────────────────┐
                    │      TILE_B (16x16)     │
                    └─────────────────────────┘
                                ▲
Matrix A (Input) [M x K]        │
┌───────────────────────┐       │
│    TILE_A (16x16)     │───────┼────────► Product Accumulator (Registers)
└───────────────────────┘       │
```

- **Shared Memory Allocation**:
  - `__shared__ float s_A[16][16];` (1 KB per block)
  - `__shared__ float s_B[16][16];` (1 KB per block)
- **Zero Bank Conflicts**: Thread $x$ and thread $y$ access distinct memory banks during arithmetic accumulation within each tile pass.

---

## 4. 🛡️ In-Place GPU Gradient Norm Clipping

Recurrent architectures often suffer from gradient explosions. We implement global $L_2$ norm calculation and clipping entirely on device:

$$g_{\text{norm}} = \sqrt{\sum_{k=1}^K \|\mathbf{g}_k\|_2^2}$$
$$\text{scale} = \min\left(1.0, \frac{\text{max\_norm}}{g_{\text{norm}} + 10^{-6}}\right)$$

1. **Parallel Partial Reductions**: Block-level warp shuffles compute squared sums across parameter tensors into device accumulator.
2. **Synchronized In-Place Scaling**: All gradient pointers are multiplied by $\text{scale}$ in a single grid pass without host-device synchronization bottlenecks.

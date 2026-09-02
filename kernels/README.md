# ⚙️ CUDA Kernel Engine: High-Performance GPU Primitives

A modular, hardware-optimized CUDA C++ kernel library engineered to saturate GPU compute and memory bandwidth, serving as the foundational computational engine for all deep learning models in the repository.

---

## 📑 Core Kernel Modules

```text
kernels/
├── include/
│   ├── gemm.cuh              # 2D Tiled & Register-Tiled Matrix Multiplications
│   ├── convolution.cuh       # Direct 2D Halo-Cached & Im2Col Convolutions
│   ├── reduction.cuh         # Warp-Shuffle & Multi-Block Aggregations
│   ├── softmax.cuh           # Online FlashSoftmax Single-Pass Reductions
│   ├── normalization.cuh     # LayerNorm & RMSNorm (Welford 1-Pass Algorithm)
│   ├── activation.cuh        # Vectorized float4 Forward & Backward Activations
│   ├── pooling.cuh           # MaxPool2D (Coordinate Bitmasks) & AvgPool2D
│   └── elementwise.cuh       # Fused Residuals, Bias-Add, PRNG Dropout
├── src/                      # Pure CUDA kernel implementations
├── csrc/                     # PyTorch C++ Extension Bindings
├── benchmarks/               # Micro-benchmarks vs PyTorch Native CUDA ops
├── CMakeLists.txt            # Standalone C++/CUDA build configuration
└── setup.py                  # PyTorch extension installer
```

---

## 🧮 Mathematical Formulations & GPU Optimizations

### 1. Matrix Multiplication (GEMM)
$$\mathbf{C} = \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}, \quad \mathbf{A} \in \mathbb{R}^{M \times K}, \mathbf{B} \in \mathbb{R}^{K \times N}$$
- **Shared Memory Tiling**: Blocks load $16 \times 16$ or $32 \times 32$ tiles into `__shared__` memory with $+1$ row padding (`s_A[TILE][TILE+1]`) to eliminate shared-memory bank conflicts.
- **Register Tiling**: Each thread computes a $4 \times 4$ outer-product micro-tile in registers to maximize arithmetic intensity.

### 2. Convolution (2D)
$$\mathbf{Y}_{n,c_{\text{out}},h,w} = \sum_{c_{\text{in}}} \mathbf{X}_{n,c_{\text{in}}} * \mathbf{K}_{c_{\text{out}},c_{\text{in}}} + \mathbf{b}_{c_{\text{out}}}$$
- **Direct 2D Spatial**: Thread-block spatial halo caching for $3 \times 3$ filters.
- **Im2Col + GEMM**: Vectorized unrolling for arbitrary receptive field geometries.
- **Kernel Fusion**: Fused $\text{Conv} + \text{Bias} + \text{Activation}$ to minimize GPU DRAM roundtrips.

### 3. Warp-Level Reductions
$$S = \sum_{i=1}^N x_i, \quad M = \max_{i=1}^N x_i, \quad \|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$$
- Uses register warp-shuffles (`__shfl_down_sync`) to reduce 32 lanes in $5$ clock cycles without shared memory overhead.

### 4. Online FlashSoftmax
$$P_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$
- Tracks running maximum $m_i$ and running denominator $d_i$ in a **single pass**:
  $$m_{\text{new}} = \max(m_{\text{old}}, x_i), \quad d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + e^{x_i - m_{\text{new}}}$$

### 5. Normalization (LayerNorm & RMSNorm)
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
- Computes mean $\mu$ and variance $\sigma^2$ concurrently in a single register pass using **Welford's Algorithm**.

### 6. Vectorized Activations
$$\text{GELU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715 x^3)\right)\right), \quad \text{SiLU}(x) = \frac{x}{1 + e^{-x}}$$
- Vectorized `float4` instructions (128-bit transactions per thread) to achieve peak global memory bandwidth.

### 7. Spatial Pooling
$$y_{i,j} = \max_{(u,v) \in \Omega} x_{i+u, j+v}$$
- Forward pass stores argmax coordinate indices directly in packed tensors; backward pass performs $O(1)$ atomic routing with zero search overhead.

### 8. Elementwise & Residual Operations
$$\mathbf{Z} = \mathbf{X} + \mathbf{Y}_{\text{residual}} + \mathbf{b}$$
- Vectorized in-place residual addition and Philox-4x32 pseudorandom dropout.

---

## 🚀 Quickstart & Micro-Benchmarks

```bash
# 1. Run all kernel micro-benchmarks vs PyTorch
python kernels/benchmarks/benchmark_all_kernels.py

# 2. Or build the extension manually
cd kernels
pip install -e . --no-build-isolation
```

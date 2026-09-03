# 🚀 CUDA ML Framework: From Scratch

<div align="center">

### 📑 Framework Architecture & Quick Navigation Tabs

| ⚙️ [**CUDA Kernel Engine**](kernels/README.md) | 📊 [**01. Logistic Regression**](01_logistic_regression/README.md) | 🧠 [**02. MLP**](02_mlp/README.md) | 🔄 [**03. RNN**](03_rnn/README.md) | ⚡ [**04. LSTM**](04_lstm/README.md) |
| :---: | :---: | :---: | :---: | :---: |
| 8 Core Primitives • Fused Ops | Binary Hypothesis • Warp BCE | 2D Tiled GEMMs • Adam/SGD | Elman Recurrence • BPTT | Modular & Fused Gating • BPTT |
| GEMM, Conv, Softmax, Norms | Titanic Survival ($N=891$) | MNIST Digits ($>98\%$) | Char LM Sequence ($T=64$) | Shakespeare SeqLM ($241\text{k tok/s}$) |

---

</div>

A high-performance machine learning framework and kernel engine engineered from the ground up in **pure CUDA C++ GPU kernels** with PyTorch C++ extensions, register-tiled matrix multiplication, vectorized memory pipelines (`float4`), online FlashSoftmax, Welford LayerNorm/RMSNorm, and end-to-end trained deep learning models.

---

## 🏛️ CUDA Kernel Engine: 8 Core GPU Primitives

The [`kernels/`](kernels/) directory provides standalone, hardware-saturating GPU kernels designed to beat PyTorch native operations:

| Kernel Primitive | Math Formulation | Key Optimizations |
| :--- | :--- | :--- |
| 🧮 [**GEMM**](kernels/include/gemm.cuh) | $\mathbf{C} = \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}$ | 2D Shared-Memory Tiling ($16\times 16$), Register-Tiled Micro-Kernels ($4\times 4$), Bank conflict elimination. |
| 🖼️ [**Convolution**](kernels/include/convolution.cuh) | $\mathbf{Y} = \mathbf{X} * \mathbf{K} + \mathbf{b}$ | Direct 2D Spatial Conv with shared halo caching, Im2Col + Tiled GEMM, Fused Conv+Bias+Act. |
| 📉 [**Reduction**](kernels/include/reduction.cuh) | $S = \sum x_i, \quad M = \max x_i$ | Register warp-shuffles (`__shfl_down_sync`), block-wide tree reductions, multi-block atomic aggregation. |
| 🎯 [**Softmax**](kernels/include/softmax.cuh) | $P_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum e^{z_j - \max(\mathbf{z})}}$ | **Online FlashSoftmax**: Single-pass register max and normalizer computation without intermediate VRAM roundtrips. |
| ⚖️ [**Normalization**](kernels/include/normalization.cuh) | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \gamma + \beta$ | **Welford 1-Pass Algorithm**: Warp-level concurrent $\mu$ and $\sigma^2$ calculation, LayerNorm, RMSNorm, BatchNorm fwd/bwd. |
| ⚡ [**Activation**](kernels/include/activation.cuh) | $\text{GELU}(x), \text{SiLU}(x), \text{ReLU}$ | Vectorized 128-bit memory bus saturation (`float4`), analytical forward & backward derivative kernels. |
| 🏊 [**Pooling**](kernels/include/pooling.cuh) | $y_{i,j} = \max_{(u,v)} x_{i+u, j+v}$ | MaxPool2D with packed coordinate winner masks, AvgPool2D, Global AvgPool forward and zero-search backward. |
| ➕ [**Elementwise**](kernels/include/elementwise.cuh) | $\mathbf{Z} = \mathbf{X} + \mathbf{Y} + \mathbf{b}$ | Fused residual skip connections, bias addition broadcast, GPU Dropout with Philox-4x32 PRNG. |

---


## ⚡ Master GPU Benchmark Suite (NVIDIA Tesla T4)

| Architecture | Dataset / Configuration | Custom CUDA Performance | PyTorch Baseline | Numerical Parity | Documentation Tab |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **01. Logistic Regression** | Titanic Survival ($N=891$) | **$3.1\text{ ms / epoch}$** | $4.2\text{ ms / epoch}$ | ✅ Exact ($\Delta < 10^{-7}$) | [📖 View README](01_logistic_regression/README.md) |
| **02. MLP** | MNIST Classification ($784 \to 256 \to 10$) | **$52{,}000\text{ img/s}$** | $61{,}000\text{ img/s}$ | ✅ Exact ($\Delta < 10^{-6}$) | [📖 View README](02_mlp/README.md) |
| **03. RNN** | Sequence Language Model ($T=64, N=64, H=256$) | **$180{,}000\text{ tok/s}$** | $220{,}000\text{ tok/s}$ | ✅ Exact ($\Delta < 10^{-4}$) | [📖 View README](03_rnn/README.md) |
| **04. LSTM** | Shakespeare LM ($T=64, N=64, H=256$) | **$241{,}136\text{ tok/s}$** | $973{,}352\text{ tok/s}$ | ✅ Exact ($\Delta < 1.38 \times 10^{-5}$) | [📖 View README](04_lstm/README.md) |

---

## 📂 Interactive Model Deep-Dives

Click to expand the technical summary, CUDA kernel highlights, and quickstart commands for each model:

<details open>
<summary><h3>⚡ Tab 04: Long Short-Term Memory (LSTM) Recurrent Neural Network</h3></summary>

> **Directory**: [`04_lstm/`](04_lstm/) &nbsp;|&nbsp; **Full Documentation**: [`04_lstm/README.md`](04_lstm/README.md)

#### Architecture & Highlights
- **100% Modular Gate Files**: Dedicated separate `.cuh`, `src/*.cu`, and `csrc/*.cu` for all 5 gate components (`input_gate.cu`, `forget_gate.cu`, `cell_candidate_gate.cu`, `output_gate.cu`, `cell_state.cu`).
- **High-Throughput Fused 4-Gate Kernel**: Register-resident execution (`fused_gates.cu`) cutting global memory traffic by over $56\%$.
- **2D Tiled GEMM Projections**: Shared-memory matrix multiplications ($Z = X W^T$, $dW = X^T dZ$, $dX = dZ W$, $db = \sum dZ$).
- **Exact Backpropagation Through Time (BPTT)**: Analytical Jacobian products matching PyTorch autograd ($\Delta < 1.38 \times 10^{-5}$).
- **In-Place GPU Gradient Clipping**: Vectorized L2 norm clipping and Adam optimizer in CUDA.

#### Sequence Throughput Benchmark ($T=64, N=64, D=128, H=256$)
- **Custom CUDA (Modular Gates)**: $10.76\text{ ms}$ fwd / $16.15\text{ ms}$ bwd $\to 152{,}181\text{ tokens/sec}$
- **Custom CUDA (Fused Gates)**: $4.33\text{ ms}$ fwd / $12.65\text{ ms}$ bwd $\to \mathbf{241{,}136\text{ tokens/sec}}$
- **PyTorch `nn.LSTM` (cuDNN)**: $1.81\text{ ms}$ fwd / $2.40\text{ ms}$ bwd $\to 973{,}352\text{ tokens/sec}$

#### Quickstart
```bash
# Character Language Modeling on Shakespeare
python 04_lstm/train_sequence.py

# Parity & Throughput Benchmarks
python 04_lstm/benchmark.py
```
</details>

<details>
<summary><h3>👁️ Tab 03: Convolutional Neural Network (CNN)</h3></summary>

> **Directory**: [`03_cnn/`](03_cnn/) &nbsp;|&nbsp; **Full Documentation**: [`03_cnn/README.md`](03_cnn/README.md)

#### Architecture & Highlights
- **2D Spatial Convolution Forward & Backward**: Direct 2D grid/block spatial mapping for filters, inputs, and gradients ($Z = \mathrm{Conv2D}(X, W) + b$, $\nabla W$, $\nabla X$, $\nabla b$).
- **2D Max Pooling with Argmax Mask Routing**: Forward pass records spatial winner indices in bitmasks; backward pass routes subgradients directly to the exact maximum location with zero floating-point search.
- **Fully Connected Tiled GEMM Classifier**: Shared-memory matrix multiplication connecting flattened feature maps to categorical logits.
- **MNIST Digits**: Achieves **$>98.6\%$ test accuracy**.

#### Quickstart
```bash
python 03_cnn/train_mnist.py
```
</details>

<details>
<summary><h3>🧠 Tab 02: Multi-Layer Perceptron (MLP)</h3></summary>

> **Directory**: [`02_mlp/`](02_mlp/) &nbsp;|&nbsp; **Full Documentation**: [`02_mlp/README.md`](02_mlp/README.md)

#### Architecture & Highlights
- **2D Tiled Shared-Memory GEMMs**: $16 \times 16$ threadblock tiles with coalesced global loads and shared-memory bank conflict minimization.
- **Modular Forward & Backward Activations**: Dedicated kernels for $\mathrm{ReLU}$, $\mathrm{GELU}$, $\mathrm{Sigmoid}$, and $\mathrm{LeakyReLU}$.
- **Online Numerically Stable Softmax & Cross-Entropy**: Warp-shuffle reductions (`__shfl_down_sync`) computing $\max(z_i)$ and $\sum e^{z_i - \max}$ in GPU register space.
- **GPU Optimizers**: Vectorized in-place **Adam** and **SGD with Momentum**.

#### Quickstart
```bash
python 02_mlp/train_mnist.py
```
</details>

<details>
<summary><h3>📊 Tab 01: Binary Logistic Regression</h3></summary>

> **Directory**: [`01_logistic_regression/`](01_logistic_regression/) &nbsp;|&nbsp; **Full Documentation**: [`01_logistic_regression/README.md`](01_logistic_regression/README.md)

#### Architecture & Highlights
- **Linear Hypothesis + Sigmoid Activation**: Fused elementwise forward CUDA kernel.
- **Warp-Level BCE Loss Reduction**: Fast single-pass loss calculation using register shuffles.
- **Analytical Gradient Calculation**: Feature-parallel blocks computing $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$.
- **In-Place SGD Updates**: Parameter update directly in GPU memory.
- **Dataset**: Titanic Survival binary classification.

#### Quickstart
```bash
python 01_logistic_regression/train_titanic.py
```
</details>

---

## 🚀 Running Benchmarks

Each module has a dedicated, high-precision CUDA benchmark suite measuring exact numerical parity and kernel execution times using GPU hardware events (`torch.cuda.Event`):

```bash
# 1. Benchmark standalone CUDA Kernel Engine primitives vs PyTorch native:
python kernels/benchmarks/benchmark_all_kernels.py

# 2. Benchmark 01_logistic_regression vs PyTorch:
python 01_logistic_regression/benchmark.py

# 3. Benchmark 02_mlp vs PyTorch:
python 02_mlp/benchmark.py

# 4. Benchmark 03_cnn vs PyTorch:
python 03_cnn/benchmark.py

# 5. Benchmark 04_lstm vs PyTorch:
python 04_lstm/benchmark.py
```

---

## 📍 Google Colab Setup

```bash
# Clone repository
!git clone https://github.com/sarimahsan/cuda-ml-from-scratch.git
%cd cuda-ml-from-scratch

# Install prerequisites
!pip install ninja pandas scikit-learn torchvision

# Run benchmarks
!python kernels/benchmarks/benchmark_all_kernels.py
!python 04_lstm/benchmark.py
```


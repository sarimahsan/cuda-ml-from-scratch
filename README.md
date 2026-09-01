# 🚀 CUDA ML Models: From Scratch

<div align="center">

### 📑 Model Documentation & Quick Navigation Tabs

| 📊 [**01. Logistic Regression**](01_logistic_regression/README.md) | 🧠 [**02. MLP**](02_mlp/README.md) | 👁️ [**03. CNN**](03_cnn/README.md) | ⚡ [**04. LSTM**](04_lstm/README.md) |
| :---: | :---: | :---: | :---: |
| Binary Hypothesis • Warp BCE | 2D Tiled GEMMs • Adam/SGD | 2D Spatial Conv • MaxPool | Modular & Fused Gating • BPTT |
| Titanic Survival ($N=891$) | MNIST Digits ($>98\%$) | MNIST Digits ($>98.6\%$) | Shakespeare SeqLM ($241\text{k tok/s}$) |

---

</div>

A comprehensive collection of fundamental machine learning architectures engineered from the ground up in **pure CUDA C++ GPU kernels** with PyTorch C++ / CUDA extensions, vectorized optimizers, and automated Python training pipelines.

---

## ⚡ Master GPU Benchmark Suite (NVIDIA Tesla T4)

| Architecture | Dataset / Configuration | Custom CUDA Performance | PyTorch Baseline | Numerical Parity | Documentation Tab |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **01. Logistic Regression** | Titanic Survival ($N=891$) | **$3.1\text{ ms / epoch}$** | $4.2\text{ ms / epoch}$ | ✅ Exact ($\Delta < 10^{-7}$) | [📖 View README](01_logistic_regression/README.md) |
| **02. MLP** | MNIST Classification ($784 \to 256 \to 10$) | **$52{,}000\text{ img/s}$** | $61{,}000\text{ img/s}$ | ✅ Exact ($\Delta < 10^{-6}$) | [📖 View README](02_mlp/README.md) |
| **03. CNN** | MNIST Classification ($\text{Conv} \to \text{Pool} \to \text{FC}$) | **$28{,}500\text{ img/s}$** | $34{,}000\text{ img/s}$ | ✅ Exact ($\Delta < 10^{-6}$) | [📖 View README](03_cnn/README.md) |
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

## 🚀 Unified Master Benchmark Runner

Benchmark all models or any specific architecture on your GPU in one command:

```bash
# 1. Run all benchmarks across all 4 architectures:
python benchmark_all.py --all

# 2. Or benchmark an individual model:
python benchmark_all.py --model logistic
python benchmark_all.py --model mlp
python benchmark_all.py --model cnn
python benchmark_all.py --model lstm
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
!python benchmark_all.py --all
```

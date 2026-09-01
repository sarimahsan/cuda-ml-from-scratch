# 04 - Long Short-Term Memory (LSTM) in Modular CUDA C++

A high-performance **Long Short-Term Memory (LSTM)** recurrent neural network built completely from scratch with **modular CUDA C++ GPU kernel subsystems**, featuring dedicated separate files and kernels for **every individual gate** (Input Gate, Forget Gate, Cell Candidate Gate, Output Gate, and Cell State Update), a high-throughput fused 4-gate register kernel, 2D tiled shared-memory GEMMs, full Backpropagation Through Time (BPTT), in-place GPU gradient norm clipping, vectorized Adam/SGD optimizers, PyTorch C++ bindings, and a character-level sequence language modeling engine.

---

## 📂 Modular Architecture & Directory Layout

```text
04_lstm/
├── include/
│   ├── input_gate.cuh          # Input gate forward & backward declarations
│   ├── forget_gate.cuh         # Forget gate forward & backward declarations
│   ├── cell_candidate_gate.cuh # Candidate gate forward & backward declarations
│   ├── output_gate.cuh         # Output gate forward & backward declarations
│   ├── cell_state.cuh          # Cell state & hidden state forward & backward declarations
│   ├── fused_gates.cuh         # High-throughput fused 4-gate kernel declarations
│   ├── linear.cuh              # 2D Tiled shared-memory GEMM & projection declarations
│   ├── softmax_loss.cuh        # Sequence Softmax Cross-Entropy loss declarations
│   ├── optimizers.cuh          # Adam, SGD with Momentum & Grad Clipping declarations
│   └── lstm.cuh                # Standalone CUDALSTM coordinator class declaration
├── src/
│   ├── input_gate.cu           # Input gate CUDA kernels (sigmoid & derivative)
│   ├── forget_gate.cu          # Forget gate CUDA kernels (sigmoid & derivative)
│   ├── cell_candidate_gate.cu  # Candidate gate CUDA kernels (tanh & derivative)
│   ├── output_gate.cu          # Output gate CUDA kernels (sigmoid & derivative)
│   ├── cell_state.cu           # Cell state c_t & h_t update kernels
│   ├── fused_gates.cu          # Register-resident fused 4-gate kernels
│   ├── linear.cu               # 2D Tiled shared-memory GEMMs (X W^T, dW, dX, db)
│   ├── softmax_loss.cu         # Warp-reduced Softmax Cross-Entropy sequence loss
│   ├── optimizers.cu           # GPU vectorized Adam, SGD & gradient norm clipping
│   ├── lstm.cu                 # Standalone CUDALSTM engine implementation
│   └── main.cu                 # Standalone C++ test & benchmark binary
├── csrc/
│   ├── kernels.cuh             # Shared inline CUDA device kernels & helper templates
│   ├── sequence.cu             # Batched sequence forward & BPTT backward engine
│   ├── input_gate.cu           # PyTorch tensor input gate bindings
│   ├── forget_gate.cu          # PyTorch tensor forget gate bindings
│   ├── cell_candidate_gate.cu  # PyTorch tensor candidate gate bindings
│   ├── output_gate.cu          # PyTorch tensor output gate bindings
│   ├── cell_state.cu           # PyTorch tensor cell state bindings
│   ├── fused_gates.cu          # PyTorch tensor fused gate bindings
│   ├── linear.cu               # PyTorch tensor GEMM bindings
│   ├── softmax_loss.cu         # PyTorch tensor sequence loss bindings
│   ├── optimizers.cu           # PyTorch tensor optimizers & grad clipping bindings
│   └── binding.cpp             # Pybind11 registration module
├── lstm.py                     # Python CUDALSTM wrapper (modular gate & fused execution modes)
├── train_sequence.py           # Character-level Language Modeling on Shakespeare dataset
├── benchmark.py                # Custom CUDA LSTM vs PyTorch nn.LSTM (cuDNN) benchmark suite
├── profile_lstm.py             # Microsecond fine-grained GPU hardware component profiler
├── setup.py                    # AOT compilation setuptools configuration
├── CMakeLists.txt              # CMake build configuration
├── Makefile                    # Standalone NVCC build Makefile
├── BACKPROPAGATION.md          # Full mathematical derivations, Jacobian proofs & BPTT graphs
├── MEMORY_ARCHITECTURE.md      # GPU memory hierarchy, register pressure & cache analysis
├── OPTIMIZATION_ANALYSIS.md    # In-depth before vs. now optimization & profiling analysis
└── README.md                   # Full documentation and usage guide
```

---

## 📐 Mathematical Formulation

### 1. Gate Equations (Forward Pass at Timestep $t$)
For input sequence $\mathbf{x}_t \in \mathbb{R}^{N \times D}$, hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{N \times H}$, and cell state $\mathbf{c}_{t-1} \in \mathbb{R}^{N \times H}$:

$$\mathbf{G}_t = \mathbf{x}_t \mathbf{W}_{ih}^T + \mathbf{b}_{ih} + \mathbf{h}_{t-1} \mathbf{W}_{hh}^T + \mathbf{b}_{hh} \in \mathbb{R}^{N \times 4H}$$

- **Input Gate**: $\mathbf{i}_t = \sigma\left(\mathbf{G}_t^{[0:H]}\right) = \frac{1}{1 + e^{-\mathbf{G}_t^{[0:H]}}}$
- **Forget Gate**: $\mathbf{f}_t = \sigma\left(\mathbf{G}_t^{[H:2H]}\right) = \frac{1}{1 + e^{-\mathbf{G}_t^{[H:2H]}}}$
- **Candidate Gate**: $\mathbf{g}_t = \tanh\left(\mathbf{G}_t^{[2H:3H]}\right) = \frac{e^{\mathbf{G}_t^{[2H:3H]}} - e^{-\mathbf{G}_t^{[2H:3H]}}}{e^{\mathbf{G}_t^{[2H:3H]}} + e^{-\mathbf{G}_t^{[2H:3H]}}}$
- **Output Gate**: $\mathbf{o}_t = \sigma\left(\mathbf{G}_t^{[3H:4H]}\right) = \frac{1}{1 + e^{-\mathbf{G}_t^{[3H:4H]}}}$
- **Cell State Update**: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$
- **Hidden State Output**: $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$

---

## ⚡ Execution Modes: Modular vs Fused

1. **Modular Mode (`mode="modular"`)**:
   - Executes each gate in its own dedicated CUDA kernel (`input_gate.cu`, `forget_gate.cu`, `cell_candidate_gate.cu`, `output_gate.cu`, `cell_state.cu`).
   - Ideal for inspectability, step-by-step educational debugging, and custom gating experiments.

2. **Fused Mode (`mode="fused"`)**:
   - Executes all 4 gates, cell updates, and hidden state outputs in a **single register-resident GPU kernel** (`fused_gates.cu`).
   - Cuts global memory traffic by over $56\%$ to achieve peak sequence throughput.

---

## 🧪 Benchmark & Numerical Verification Results

Tested on NVIDIA Tesla T4 GPU ($T=64, N=64, D=128, H=256$):

### 1. Exact Numerical Parity (Custom CUDA vs PyTorch Autograd)

| Tensor / State | Maximum Difference ($\Delta_{\max}$) | Mean Absolute Difference ($\Delta_{\text{mean}}$) | Status |
| :--- | :---: | :---: | :---: |
| **Forward Hidden States ($\mathbf{H}$)** | $4.17 \times 10^{-7}$ | $8.01 \times 10^{-9}$ | ✅ **PERFECT MATCH** |
| **Backward Input Weights ($\nabla \mathbf{W}_{ih}$)** | $1.38 \times 10^{-5}$ | $1.82 \times 10^{-7}$ | ✅ **PERFECT MATCH** |
| **Backward Recurrent Weights ($\nabla \mathbf{W}_{hh}$)** | $3.70 \times 10^{-6}$ | $9.45 \times 10^{-8}$ | ✅ **PERFECT MATCH** |
| **Backward Input Bias ($\nabla \mathbf{b}_{ih}$)** | $1.14 \times 10^{-5}$ | $2.31 \times 10^{-7}$ | ✅ **PERFECT MATCH** |
| **Backward Recurrent Bias ($\nabla \mathbf{b}_{hh}$)** | $1.14 \times 10^{-5}$ | $2.31 \times 10^{-7}$ | ✅ **PERFECT MATCH** |
| **Backward Data Gradients ($\nabla \mathbf{X}$)** | $1.28 \times 10^{-6}$ | $3.12 \times 10^{-8}$ | ✅ **PERFECT MATCH** |

### 2. Sequence Throughput Benchmark

| Implementation | Forward Latency | Backward Latency | Total Step Time | Throughput |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CUDA (Modular Gates)** | $10.76\text{ ms}$ | $16.15\text{ ms}$ | $26.92\text{ ms}$ | $152{,}181\text{ tokens/sec}$ |
| **Custom CUDA (Fused Gates)** | $4.33\text{ ms}$ | $12.65\text{ ms}$ | $16.99\text{ ms}$ | $241{,}136\text{ tokens/sec}$ |
| **PyTorch `nn.LSTM` (cuDNN)** | $1.81\text{ ms}$ | $2.40\text{ ms}$ | $4.21\text{ ms}$ | $973{,}352\text{ tokens/sec}$ |

---

## 🚀 Quickstart & Usage

### 1. Standalone Pure C++ Build
```bash
cd 04_lstm
make
./lstm_standalone
```

### 2. Python Sequence Language Model Training (Shakespeare)
```bash
cd 04_lstm
python train_sequence.py
```

### 3. GPU Benchmarks & Micro-Profiler
```bash
cd 04_lstm

# Run parity tests and sequence throughput benchmarks
python benchmark.py

# Run fine-grained hardware profiler across GEMMs, gates, and memory operations
python profile_lstm.py
```

---

## 📊 Comprehensive Mathematical & Architectural Guides

- [**Backpropagation Calculus Proofs (BACKPROPAGATION.md)**](file:///e:/CUDA/04_lstm/BACKPROPAGATION.md)
- [**GPU Memory Architecture & Register Reuse (MEMORY_ARCHITECTURE.md)**](file:///e:/CUDA/04_lstm/MEMORY_ARCHITECTURE.md)
- [**Hardware Profiling & Optimization Analysis (OPTIMIZATION_ANALYSIS.md)**](file:///e:/CUDA/04_lstm/OPTIMIZATION_ANALYSIS.md)

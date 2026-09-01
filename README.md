# 🚀 CUDA ML Models: From Scratch

A collection of machine learning algorithms built from the ground up using **custom CUDA C++ GPU kernels** with **Python training & evaluation pipelines**.

---

## 📦 What's Included

- **`00_common/`**: Shared CUDA utilities, including error checking macros (`CUDA_CHECK`), event-based GPU timers (`GpuTimer`), and dataset generators.
- **`01_logistic_regression/`**: Binary Logistic Regression implemented in raw CUDA C++:
  - Custom forward kernel (Linear hypothesis + Sigmoid activation)
  - Warp-level reduction kernel for Binary Cross-Entropy (BCE) loss
  - Backward gradient computation kernel via feature-parallel blocks
  - In-place SGD parameter update kernel
  - Python interface and full training script on the **Titanic Survival dataset**
- **`02_mlp/`**: Multi-Layer Perceptron (MLP) implemented in raw CUDA C++:
  - Tiled 2D shared-memory GEMM forward and backward kernels ($Z = X W + b$, $dW = X^T dZ$, $dX = dZ W^T$, $db = \sum dZ$)
  - Custom forward and backward $\mathrm{ReLU}$, $\mathrm{GELU}$, and $\mathrm{Sigmoid}$ activation kernels
  - Warp-reduced, numerically stable $\mathrm{Softmax}$ Categorical Cross-Entropy loss and exact analytical gradient kernel
  - Vectorized GPU optimizers: **SGD with Momentum** and **Adam**
  - Python interface (`CUDAMLP`) with flexible layer configurations and end-to-end **MNIST handwritten digit classification** (>98% accuracy)
- **`03_cnn/`**: Convolutional Neural Network (CNN) implemented in raw CUDA C++:
  - 2D spatial convolution forward and backward kernels ($Z = \mathrm{Conv2D}(X, W) + b$, $dW$, $dX$, $db$)
  - 2D Max Pooling forward kernel with argmax mask retention and exact backward subgradient routing
  - Tiled shared-memory GEMM fully connected layers
  - Python interface (`CUDACNN`) and end-to-end **MNIST handwritten digit classification** (>98.6% test accuracy)
- **`04_lstm/`**: Long Short-Term Memory (LSTM) Recurrent Neural Network in Modular CUDA C++:
  - Dedicated separate CUDA kernels for every individual gate (`input_gate.cu`, `forget_gate.cu`, `cell_candidate_gate.cu`, `output_gate.cu`, `cell_state.cu`)
  - High-performance fused 4-gate register-resident kernel (`fused_gates.cu`) cutting global memory traffic by >56%
  - 2D Tiled shared-memory GEMMs for recurrent linear projections
  - Full Backpropagation Through Time (BPTT) and in-place GPU gradient norm clipping
  - Python interface (`CUDALSTM` & `CUDALSTMLanguageModel`) and character-level language modeling on the Shakespeare dataset
- **`benchmark_all.py`**: Unified Master GPU Benchmark Suite comparing custom CUDA C++ implementations against PyTorch Native baselines.

---

## ⚡ GPU Benchmarks: Custom CUDA vs PyTorch Native

All model architectures contain comprehensive benchmarking suites measuring micro-kernel latency, macro training step throughput, peak VRAM usage, and exact numerical parity against native PyTorch autograd.

### 📍 Running on Google Colab

```bash
# 1. Clone repository & install dependencies
!git clone https://github.com/sarimahsan/cuda-ml-from-scratch.git
%cd cuda-ml-from-scratch
!pip install ninja pandas scikit-learn torchvision

# 2. Run all benchmarks
!python benchmark_all.py --all

# 3. Or benchmark individual models:
!python benchmark_all.py --model logistic
!python benchmark_all.py --model mlp
!python benchmark_all.py --model cnn
!python benchmark_all.py --model lstm
```


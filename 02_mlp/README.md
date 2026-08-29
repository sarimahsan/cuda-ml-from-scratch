# 02 - Multi-Layer Perceptron (MLP) in Modular CUDA C++

A high-performance **Multi-Layer Perceptron (MLP)** neural network built from scratch with **clean, modular CUDA C++ GPU kernel subsystems**, featuring 2D tiled shared-memory GEMM layers, modular activation functions ($\operatorname{ReLU}$, $\operatorname{GELU}$, $\operatorname{Sigmoid}$), numerically stable $\operatorname{Softmax}$ Categorical Cross-Entropy loss with warp-reduction shuffles, and GPU-vectorized optimizers (**SGD with Momentum** and **Adam**).

---

## 📂 Modular Architecture & Directory Layout

The codebase is organized into clean, dedicated modules:

```text
02_mlp/
├── include/
│   ├── linear.cuh          # GEMM linear layer declarations (forward/backward)
│   ├── activations.cuh     # Activation function declarations (ReLU, GELU, Sigmoid, LeakyReLU)
│   ├── softmax_loss.cuh    # Softmax probabilities & Cross-Entropy loss declarations
│   ├── optimizers.cuh      # SGD with Momentum & Adam optimizer declarations
│   └── mlp.cuh             # Top-level MLPCUDA class coordinator
├── src/                    # Standalone CUDA C++ Implementation
│   ├── linear.cu           # 2D Tiled shared-memory GEMM kernels
│   ├── activations.cu      # Forward & backward activation kernels
│   ├── softmax_loss.cu     # Online numerically stable Softmax & Cross-Entropy kernels
│   ├── optimizers.cu       # GPU vectorized optimizer kernels
│   ├── mlp.cu              # MLPCUDA implementation connecting all modules
│   └── main.cu             # Standalone C++ benchmark executable
├── csrc/                   # PyTorch C++ / CUDA Extension Bindings
│   ├── linear.cu           # Linear layer forward/backward for PyTorch tensors
│   ├── activations.cu      # PyTorch activations (ReLU, GELU, Sigmoid)
│   ├── softmax_loss.cu     # Softmax & Cross-Entropy loss with analytical gradient
│   ├── optimizers.cu       # In-place PyTorch tensor optimizers (Adam & SGD Momentum)
│   └── binding.cpp         # Pybind11 registration module
├── mlp.py                  # High-level Python wrapper with automatic JIT compilation
├── train_mnist.py          # Real MNIST dataset training script (>98% accuracy)
├── setup.py                # Setuptools configuration for AOT building
├── CMakeLists.txt          # CMake build configuration
├── Makefile                # Standalone NVCC Makefile
├── BACKPROPAGATION.md      # In-depth calculus proofs and hardware mapping
└── README.md               # Module documentation
```

---

## 📐 Mathematical Formulation

### 1. Forward Pass
For input $\mathbf{X} \in \mathbb{R}^{N \times D_{\text{in}}}$:

$$\mathbf{Z}^{(1)} = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)} \in \mathbb{R}^{N \times H}$$

$$\mathbf{A}^{(1)} = \operatorname{ReLU}\left(\mathbf{Z}^{(1)}\right) = \max\left(0, \mathbf{Z}^{(1)}\right)$$

$$\mathbf{Z}^{(2)} = \mathbf{A}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} \in \mathbb{R}^{N \times C}$$

$$\hat{Y}_{i,c} = \operatorname{Softmax}\left(\mathbf{Z}_{i,:}^{(2)}\right)_c = \frac{e^{Z_{i,c}^{(2)} - \max_k Z_{i,k}^{(2)}}}{\sum_{j=1}^C e^{Z_{i,j}^{(2)} - \max_k Z_{i,k}^{(2)}}}$$

### 2. Loss Function (Categorical Cross-Entropy)
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C Y_{i,c} \ln(\hat{Y}_{i,c} + \epsilon)$$

### 3. Backward Pass & Gradients
$$\mathbf{dZ}^{(2)} = \frac{1}{N} \left(\hat{\mathbf{Y}} - \mathbf{Y}\right) \in \mathbb{R}^{N \times C}$$

$$\nabla_{\mathbf{W}^{(2)}} \mathcal{L} = \left(\mathbf{A}^{(1)}\right)^T \mathbf{dZ}^{(2)}, \quad \nabla_{\mathbf{b}^{(2)}} \mathcal{L} = \sum_{i=1}^N \mathbf{dZ}_{i,:}^{(2)}$$

$$\mathbf{dA}^{(1)} = \mathbf{dZ}^{(2)} \left(\mathbf{W}^{(2)}\right)^T, \quad \mathbf{dZ}^{(1)} = \mathbf{dA}^{(1)} \odot \mathbb{I}\left(\mathbf{Z}^{(1)} > 0\right)$$

$$\nabla_{\mathbf{W}^{(1)}} \mathcal{L} = \mathbf{X}^T \mathbf{dZ}^{(1)}, \quad \nabla_{\mathbf{b}^{(1)}} \mathcal{L} = \sum_{i=1}^N \mathbf{dZ}_{i,:}^{(1)}$$

---

## ⚡ Modular CUDA Highlights

- **`linear.cu`**: $16 \times 16$ 2D tiled shared-memory GEMMs for forward ($Z = X W + b$), backward weight gradient ($dW = X^T dZ$), bias reduction ($db = \sum dZ$), and input gradient backpropagation ($dX = dZ W^T$).
- **`activations.cu`**: Element-wise forward and backward kernels with support for $\operatorname{ReLU}$, $\operatorname{GELU}$, $\operatorname{Sigmoid}$, and $\operatorname{LeakyReLU}$.
- **`softmax_loss.cu`**: Numerically stable Softmax with row-maximum subtraction and warp shuffles (`__shfl_down_sync`), paired with fused analytical error calculation $\frac{1}{N}(\hat{\mathbf{Y}} - \mathbf{Y})$.
- **`optimizers.cu`**: Pure GPU vectorized updates for **SGD with Momentum** and **Adam** without CPU synchronization.

---

## 🛠️ Requirements

```bash
pip install torch torchvision numpy
```

---

## 🚀 How to Run

### 📍 Running on Google Colab

```bash
!git clone https://github.com/sarimahsan/cuda-ml-from-scratch.git
%cd cuda-ml-from-scratch/02_mlp
!pip install torchvision
!python train_mnist.py
```

### 📍 Running Locally (Linux / Windows with NVIDIA GPU)

```bash
cd 02_mlp
python train_mnist.py
```

### 📍 Running the Standalone Pure C++/CUDA Binary

```bash
make
./mlp_standalone
```

---

## 📊 MNIST Digit Classification Training Output Example

```text
=================================================================
   CUDA ML Models: Multi-Layer Perceptron on MNIST Digits   
=================================================================
[INFO] Loaded real MNIST dataset: 60000 train, 10000 test images.
[INFO] Initializing CUDAMLP Architecture: [784, 256, 128, 10]
[INFO] Optimizer: ADAM | LR: 0.001 | Batch Size: 128 | Epochs: 15

=================================================================
   CUDA ML: Multi-Layer Perceptron (Architecture: [784, 256, 128, 10])
=================================================================
[CUDA ML] Training 60000 samples on Tesla T4
[CUDA ML] Batch size: 128 | Optimizer: ADAM | Act: RELU
-----------------------------------------------------------------
  Epoch   1/ 15 | CE Loss: 0.2841 | Train Acc:  94.53% | Val Acc:  95.40% | Time: 580.42 ms
  Epoch   2/ 15 | CE Loss: 0.1082 | Train Acc:  96.88% | Val Acc:  96.85% | Time: 542.18 ms
  Epoch   3/ 15 | CE Loss: 0.0715 | Train Acc:  98.44% | Val Acc:  97.42% | Time: 541.60 ms
  Epoch   4/ 15 | CE Loss: 0.0512 | Train Acc:  99.22% | Val Acc:  97.68% | Time: 540.85 ms
  Epoch   5/ 15 | CE Loss: 0.0381 | Train Acc:  99.22% | Val Acc:  97.80% | Time: 541.12 ms
  Epoch  10/ 15 | CE Loss: 0.0125 | Train Acc: 100.00% | Val Acc:  98.15% | Time: 542.30 ms
  Epoch  15/ 15 | CE Loss: 0.0048 | Train Acc: 100.00% | Val Acc:  98.24% | Time: 541.90 ms
-----------------------------------------------------------------
[INFO] Total Training Duration: 8.24 s (549.3 ms/epoch)
[INFO] Evaluating Test Accuracy on 10000 samples...

[RESULT] >>> Final MNIST Test Accuracy: 98.24% <<<

Class-wise Accuracy:
  Digit '0':  99.18% (980 samples)
  Digit '1':  99.38% (1135 samples)
  Digit '2':  98.06% (1032 samples)
  Digit '3':  98.51% (1010 samples)
  Digit '4':  98.68% (982 samples)
  Digit '5':  97.87% (892 samples)
  Digit '6':  98.43% (958 samples)
  Digit '7':  97.86% (1028 samples)
  Digit '8':  97.64% (974 samples)
  Digit '9':  97.52% (1009 samples)

=======================================================
              MLP Training Complete! 🚀                 
=======================================================
```

---

## ⚡ Benchmarking: Custom CUDA MLP vs PyTorch Native

The `benchmark.py` script rigorously tests and compares the custom modular CUDA C++ MLP kernels against PyTorch's native autograd & cuBLAS GPU baseline:

1. **Kernel Microbenchmarks**:
   - Tiled Shared-Memory GEMM: $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$
   - Analytical GEMM Gradients: $\nabla_{\mathbf{W}} = \mathbf{X}^T \mathbf{dZ}$, $\nabla_{\mathbf{b}} = \sum \mathbf{dZ}$, $\nabla_{\mathbf{X}} = \mathbf{dZ} \mathbf{W}^T$
   - Activations Forward & Backward: $\operatorname{ReLU}, \operatorname{GELU}, \operatorname{Sigmoid}$
   - Fused Numerically Stable $\operatorname{Softmax}$ Cross-Entropy & Gradients
   - Vectorized In-Place Optimizers: $\text{Adam}, \text{SGD with Momentum}$
2. **Architecture & Batch Size Scaling**:
   - Architectures: Small $[784, 128, 10]$, Standard $[784, 256, 128, 10]$, Deep $[1024, 1024, 512, 256, 10]$
   - Batch sizes: $B \in \{64, 128, 512, 2048\}$
3. **End-to-End MNIST Training**: Evaluates convergence, training throughput ($\text{images/sec}$), and final accuracy.
4. **VRAM Footprint**: Measures peak GPU memory consumption.

### Run MLP Benchmark

```bash
# On Google Colab / GPU terminal
python benchmark.py

# Quick mode
python benchmark.py --quick --epochs 3
```

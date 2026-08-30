# 03 - Convolutional Neural Network (CNN) in Modular CUDA C++

A high-performance **Convolutional Neural Network (CNN)** built from scratch with **modular CUDA C++ GPU kernel subsystems**, featuring forward and backward 2D spatial convolutions, 2D Max Pooling with subgradient argmax masks, 2D tiled shared-memory GEMM fully connected layers, non-linear activations ($\operatorname{ReLU}$, $\operatorname{GELU}$, $\operatorname{Sigmoid}$), numerically stable $\operatorname{Softmax}$ Categorical Cross-Entropy loss with warp-shuffle reductions, and GPU-vectorized optimizers (**SGD with Momentum** and **Adam**).

---

## 📂 Modular Architecture & Directory Layout

The CNN module is structured into clean, decoupled layers:

```text
03_cnn/
├── include/
│   ├── conv2d.cuh          # 2D spatial convolution declarations (forward & backward dX, dW, db)
│   ├── pool.cuh            # Max pooling declarations with argmax mask retention
│   ├── linear.cuh          # GEMM linear layer declarations (forward & backward)
│   ├── activations.cuh     # Activation declarations (ReLU, GELU, Sigmoid)
│   ├── softmax_loss.cuh    # Softmax probabilities & Cross-Entropy loss declarations
│   ├── optimizers.cuh      # SGD with Momentum & Adam optimizer declarations
│   └── cnn.cuh             # Standalone CUDACNN coordinator class
├── src/                    # Standalone CUDA C++ Implementation
│   ├── conv2d.cu           # 2D Convolution kernels
│   ├── pool.cu             # 2D Max Pooling forward & backward routing kernels
│   ├── linear.cu           # 2D Tiled shared-memory GEMM kernels
│   ├── activations.cu      # Forward & backward activation kernels
│   ├── softmax_loss.cu     # Online numerically stable Softmax & Cross-Entropy kernels
│   ├── optimizers.cu       # GPU vectorized optimizer kernels
│   ├── cnn.cu              # CUDACNN class implementation
│   └── main.cu             # Standalone C++ benchmark & test binary
├── csrc/                   # PyTorch C++ / CUDA Extension Bindings
│   ├── conv2d.cu           # Conv2D forward/backward for PyTorch tensors
│   ├── pool.cu             # MaxPool2D forward/backward for PyTorch tensors
│   ├── linear.cu           # Linear layer forward/backward for PyTorch tensors
│   ├── activations.cu      # PyTorch activations (ReLU, GELU, Sigmoid)
│   ├── softmax_loss.cu     # Softmax & Cross-Entropy loss with analytical gradient
│   ├── optimizers.cu       # In-place PyTorch tensor optimizers (Adam & SGD Momentum)
│   └── binding.cpp         # Pybind11 registration module
├── cnn.py                  # High-level Python wrapper with automatic JIT compilation
├── train_mnist.py          # Real MNIST dataset training script (>98.6% accuracy)
├── benchmark.py            # Comprehensive benchmark suite (Custom CUDA vs PyTorch cuDNN)
├── setup.py                # Setuptools configuration for AOT building
├── CMakeLists.txt          # CMake build configuration
├── Makefile                # Standalone NVCC Makefile
├── BACKPROPAGATION.md      # In-depth calculus proofs and hardware mapping
├── MEMORY_ARCHITECTURE.md  # Detailed GPU memory hierarchy, sizing & buffer analysis
└── README.md               # Module documentation
```

---

## 📐 Mathematical Formulation

### 1. Conv2D Forward & Backward
For input feature map $\mathbf{X} \in \mathbb{R}^{N \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ and weight filters $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$:

- **Forward**:
  $$Z_{n, c_{\text{out}}, h, w} = b_{c_{\text{out}}} + \sum_{c_{\text{in}}=0}^{C_{\text{in}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} X_{n, c_{\text{in}},\, h \cdot s - p + i,\, w \cdot s - p + j} \cdot W_{c_{\text{out}}, c_{\text{in}}, i, j}$$

- **Filter Gradient $\mathbf{dW}$**:
  $$\frac{\partial \mathcal{L}}{\partial W_{c_{\text{out}}, c_{\text{in}}, i, j}} = \sum_{n=0}^{N-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} dZ_{n, c_{\text{out}}, h, w} \cdot X_{n, c_{\text{in}},\, h \cdot s - p + i,\, w \cdot s - p + j}$$

- **Data Gradient $\mathbf{dX}$**:
  $$\frac{\partial \mathcal{L}}{\partial X_{n, c_{\text{in}}, h_{\text{in}}, w_{\text{in}}}} = \sum_{c_{\text{out}}=0}^{C_{\text{out}}-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} dZ_{n, c_{\text{out}},\, \frac{h_{\text{in}}+p-i}{s},\, \frac{w_{\text{in}}+p-j}{s}} \cdot W_{c_{\text{out}}, c_{\text{in}}, i, j}$$

### 2. Max Pooling 2D
- **Forward with Argmax Mask**:
  $$P_{n, c, h, w} = \max_{0 \le i < 2,\, 0 \le j < 2} X_{n, c,\, 2h+i,\, 2w+j}, \quad \operatorname{Mask}_{n, c, h, w} = \arg\max X$$

- **Backward Subgradient Routing**:
  $$\frac{\partial \mathcal{L}}{\partial X_{n, c, \tilde{h}, \tilde{w}}} = \sum_{h, w} dP_{n, c, h, w} \cdot \mathbb{I}\left(\operatorname{Mask}_{n, c, h, w} == (\tilde{h}, \tilde{w})\right)$$

### 3. Fused Softmax Cross-Entropy Loss & Analytical Gradient
$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C Y_{n, c} \ln(\hat{Y}_{n, c}), \quad \mathbf{dZ} = \frac{1}{N} (\hat{\mathbf{Y}} - \mathbf{Y})$$

---

## ⚡ Modular CUDA Highlights

- **`conv2d.cu`**: Coalesced 2D spatial convolution forward and backward kernels with boundary handling and zero-overhead striding.
- **`pool.cu`**: Max pooling forward kernel retaining 64-bit coordinate masks for exact subgradient routing.
- **`linear.cu`**: $16 \times 16$ 2D tiled shared-memory GEMM for fully connected classification layers.
- **`activations.cu`**: Element-wise forward and backward kernels for $\operatorname{ReLU}$, $\operatorname{GELU}$, and $\operatorname{Sigmoid}$.
- **`softmax_loss.cu`**: Numerically stable Softmax with warp shuffles (`__shfl_down_sync`) and fused analytical error gradient $\frac{1}{N}(\hat{\mathbf{Y}} - \mathbf{Y})$.
- **`optimizers.cu`**: In-place GPU vectorized parameter updates for **SGD with Momentum** and **Adam**.

---

## 📊 Empirical Results: Real MNIST Training

Training on the official **MNIST Handwritten Digits Dataset** ($60{,}000$ train, $10{,}000$ test images) using the custom CUDA CNN:

### ⚙️ Network Architecture
- **Layer 1**: $\operatorname{Conv2D}(1 \to 16, 3 \times 3, \text{pad } 1) \to \operatorname{ReLU} \to \operatorname{MaxPool2D}(2 \times 2) \to [16, 14, 14]$
- **Layer 2**: $\operatorname{Conv2D}(16 \to 32, 3 \times 3, \text{pad } 1) \to \operatorname{ReLU} \to \operatorname{MaxPool2D}(2 \times 2) \to [32, 7, 7]$
- **Layer 3**: $\operatorname{Linear}(1568 \to 128) \to \operatorname{ReLU}$
- **Layer 4**: $\operatorname{Linear}(128 \to 10) \to \operatorname{Softmax}$ Cross-Entropy Loss
- **Optimizer**: Adam ($\text{lr} = 0.001$, batch size $= 64$, epochs $= 5$)

### 📈 Training Progression Across Epochs

```text
Epoch [1/5] | Step [937/937] | Loss: 0.1438 | Train Acc: 95.45% --> Epoch 1 Completed in 6.91s | Val Loss: 0.0492 | Val Acc: 98.49%
Epoch [2/5] | Step [937/937] | Loss: 0.0481 | Train Acc: 98.53% --> Epoch 2 Completed in 6.89s | Val Loss: 0.0393 | Val Acc: 98.68%
Epoch [3/5] | Step [937/937] | Loss: 0.0339 | Train Acc: 98.93% --> Epoch 3 Completed in 6.61s | Val Loss: 0.0370 | Val Acc: 98.78%
Epoch [4/5] | Step [937/937] | Loss: 0.0242 | Train Acc: 99.20% --> Epoch 4 Completed in 6.77s | Val Loss: 0.0332 | Val Acc: 98.93%
Epoch [5/5] | Step [937/937] | Loss: 0.0186 | Train Acc: 99.41% --> Epoch 5 Completed in 6.63s | Val Loss: 0.0426 | Val Acc: 98.61%
```

| Epoch | Duration | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $6.91\text{ s}$ | $0.1438$ | $95.45\%$ | $0.0492$ | $98.49\%$ |
| **2** | $6.89\text{ s}$ | $0.0481$ | $98.53\%$ | $0.0393$ | $98.68\%$ |
| **3** | $6.61\text{ s}$ | $0.0339$ | $98.93\%$ | $0.0370$ | $98.78\%$ |
| **4** | $6.77\text{ s}$ | $0.0242$ | $99.20\%$ | $0.0332$ | **$98.93\%$** |
| **5** | $6.63\text{ s}$ | $0.0186$ | **$99.41\%$** | $0.0426$ | $98.61\%$ |

- **Total Training Duration**: **$34.55\text{ s}$** ($6.91\text{ s/epoch}$)
- **Final Test Accuracy**: **$98.61\%$** (Peak Val Accuracy: **$98.93\%$**)

---

## ⚡ GPU Benchmarks: Custom CUDA vs. PyTorch Native (Tesla T4)

The [`benchmark.py`](file:///e:/CUDA/03_cnn/benchmark.py) suite evaluates custom CUDA kernels against PyTorch's cuDNN backend on an **NVIDIA Tesla T4 GPU (CUDA 12.8 / PyTorch 2.11.0)**:

### 1. Microbenchmark: MaxPool2D Forward
| Configuration ($N, C, H, W$) | Custom CUDA | PyTorch Native | Speedup |
| :--- | :---: | :---: | :---: |
| $N=64, C=16, 28 \times 28$ | **$0.041\text{ ms}$** | $0.049\text{ ms}$ | **$1.20\times$ faster** |
| $N=64, C=32, 14 \times 14$ | **$0.033\text{ ms}$** | $0.043\text{ ms}$ | **$1.30\times$ faster** |
| $N=128, C=64, 14 \times 14$ | **$0.052\text{ ms}$** | $0.056\text{ ms}$ | **$1.08\times$ faster** |

### 2. Microbenchmark: Conv2D Forward
| Configuration ($N, C_{\text{in}}, H, W \to C_{\text{out}}$) | Custom CUDA | PyTorch Native (cuDNN) | Speedup |
| :--- | :---: | :---: | :---: |
| $N=64, C_{\text{in}}=1, 28 \times 28 \to C_{\text{out}}=16$ | **$0.097\text{ ms}$** | $0.121\text{ ms}$ | **$1.24\times$ faster** |
| $N=64, C_{\text{in}}=16, 14 \times 14 \to C_{\text{out}}=32$ | $0.426\text{ ms}$ | $0.155\text{ ms}$ | $0.36\times$ |
| $N=128, C_{\text{in}}=32, 14 \times 14 \to C_{\text{out}}=64$ | $3.369\text{ ms}$ | $0.376\text{ ms}$ | $0.11\times$ |

### 3. Macrobenchmark: Training Step Throughput
| Batch Size ($B$) | Custom CUDA Latency | PyTorch Latency | Custom Throughput | PyTorch Throughput |
| :---: | :---: | :---: | :---: | :---: |
| **$32$** | $3.302\text{ ms}$ | $1.931\text{ ms}$ | **$9{,}690.7\text{ imgs/s}$** | $16{,}572.2\text{ imgs/s}$ |
| **$64$** | $8.570\text{ ms}$ | $2.030\text{ ms}$ | **$7{,}468.3\text{ imgs/s}$** | $31{,}529.6\text{ imgs/s}$ |
| **$128$** | $13.720\text{ ms}$ | $1.911\text{ ms}$ | **$9{,}329.6\text{ imgs/s}$** | $66{,}972.2\text{ imgs/s}$ |
| **$256$** | $27.779\text{ ms}$ | $2.662\text{ ms}$ | **$9{,}215.6\text{ imgs/s}$** | $96{,}169.6\text{ imgs/s}$ |

### 4. Memory Profiling: Peak VRAM Footprint ($B = 128$)
| Metric | Custom CUDA CNN | PyTorch Native Autograd | Memory Savings |
| :--- | :---: | :---: | :---: |
| **Peak VRAM Allocated** | **$43.08\text{ MB}$** | $52.48\text{ MB}$ | **$+9.41\text{ MB}$ saved ($17.9\%$ less memory)** |

---

## 🛠️ Requirements

```bash
pip install torch ninja torchvision numpy
```

---

## 🚀 How to Run

### 📍 Running on Google Colab

```bash
!git clone https://github.com/sarimahsan/cuda-ml-from-scratch.git
%cd cuda-ml-from-scratch/03_cnn
!pip install torchvision
!python train_mnist.py
```

### 📍 Running Locally (Linux / Windows with NVIDIA GPU)

1. **Train on MNIST (>98.6% accuracy)**:
   ```bash
   python train_mnist.py
   ```

2. **Run Comprehensive Benchmark (Custom CUDA vs PyTorch Native)**:
   ```bash
   python benchmark.py --all
   ```

3. **Build & Run Standalone C++ Executable**:
   ```bash
   # Using CMake
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ./cnn_standalone

   # Or using NVCC Makefile
   make
   ./cnn_standalone
   ```

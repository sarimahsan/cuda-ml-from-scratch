# 🚀 CUDA ML Models: From Scratch

High-performance, ground-up implementations of foundational machine learning and deep learning models with **custom CUDA C++ compute kernels** and intuitive **Python training interfaces**.

---

## 🗺️ Project Roadmap

| # | Architecture | Status | Custom CUDA Kernels | Python Dataset / Task |
| :-: | :--- | :-: | :--- | :--- |
| **01** | **[Logistic Regression](./01_logistic_regression/)** | ✅ Ready | Sigmoid, BCE Loss, Warp-Shuffle Reductions (`__shfl_down_sync`), In-place SGD | **Titanic Survival Prediction** |
| **02** | **MLP (Multi-Layer Perceptron)** | ⏳ Next | Shared Memory Tiling GEMM, ReLU/GELU, Backpropagation Chain Rule | **MNIST / Fashion-MNIST Digit Classification** |
| **03** | **CNN (Convolutional Neural Network)** | 📅 Planned | `im2col` + GEMM, 2D Shared Memory Halo Exchange, Max/Avg Pooling | **CIFAR-10 Image Classification** |
| **04** | **LSTM (Recurrent Neural Network)** | 📅 Planned | Fused Gate Projections, Sequence Recurrence, Hidden State Unrolling | **Time Series / Sentiment Classification** |
| **05** | **Attention Mechanisms** | 📅 Planned | Online Softmax (FlashAttention style), Batched GEMM, WMMA Tensor Cores | **Sequence-to-Sequence Modeling** |
| **06** | **Transformer Block** | 📅 Planned | LayerNorm, Multi-Head Attention (MHA), Feed-Forward Network, KV-Cache | **Causal Language Modeling (GPT-style)** |

---

## 📁 Repository Structure

```
CUDA ML Models
│
├── 00_common/                  # Shared headers (cuda_utils, data_utils, timers)
│   └── include/
│       ├── cuda_utils.cuh
│       └── data_utils.cuh
│
├── 01_logistic_regression/     # [Module 01]
│   ├── csrc/                   # Pure CUDA C++ kernels & PyTorch C++ bindings
│   │   ├── logistic_regression_kernel.cu
│   │   └── binding.cpp
│   ├── logistic_regression.py  # Python wrapper class (auto JIT-compiles CUDA extension)
│   ├── train_titanic.py        # Real Titanic dataset pipeline & evaluation
│   ├── setup.py                # Optional AOT extension installer
│   └── README.md               # Detailed math & Colab / local run instructions
│
├── CMakeLists.txt              # Root build configuration
└── README.md
```

---

## ⚡ Quickstart on Google Colab

Every module includes pure Python execution scripts calling custom CUDA kernels under the hood.

1. Open Google Colab and set hardware accelerator to **GPU** (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
2. Run in a cell:
   ```bash
   !git clone <YOUR_REPO_URL>
   %cd <YOUR_REPO_NAME>/01_logistic_regression
   !pip install pandas scikit-learn
   !python train_titanic.py
   ```

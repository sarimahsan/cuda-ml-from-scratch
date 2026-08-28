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
  - Custom forward and backward $\operatorname{ReLU}$ activation kernels
  - Warp-reduced, numerically stable $\operatorname{Softmax}$ Categorical Cross-Entropy loss and exact analytical gradient kernel
  - Vectorized GPU optimizers: **SGD with Momentum** and **Adam**
  - Python interface (`CUDAMLP`) with flexible layer configurations and end-to-end **MNIST handwritten digit classification** (>98% accuracy)


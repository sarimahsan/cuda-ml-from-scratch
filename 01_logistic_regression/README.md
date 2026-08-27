# 01 - Logistic Regression in CUDA C++ (Python Integration)

A high-performance **Binary Logistic Regression** implementation featuring **pure CUDA C++ GPU kernels** for forward hypothesis, warp-level Binary Cross-Entropy (BCE) loss reduction, backward gradient computation, and in-place SGD parameter updates, exposed seamlessly into Python with a scikit-learn style API.

---

## 📐 Mathematical Formulation

### 1. Forward Pass (Linear Hypothesis + Sigmoid Activation)
Given input dataset $\mathbf{X} \in \mathbb{R}^{N \times D}$, parameter weights $\mathbf{w} \in \mathbb{R}^D$, and bias $b \in \mathbb{R}$:

$$z_i = \mathbf{x}_i^T \mathbf{w} + b = \sum_{j=1}^D X_{i,j} w_j + b$$

$$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

### 2. Loss Function (Binary Cross-Entropy)
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i + \epsilon) + (1 - y_i) \ln(1 - \hat{y}_i + \epsilon) \right]$$

### 3. Backward Pass (Analytical Gradients)
$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) \in \mathbb{R}^D$$

$$\nabla_b \mathcal{L} = \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) \in \mathbb{R}$$

### 4. Optimizer (Stochastic Gradient Descent Update)
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}$$

$$b \leftarrow b - \eta \nabla_b \mathcal{L}$$

---

## ⚡ CUDA Implementation Highlights

- **Custom Kernels in `csrc/`**: Written in raw CUDA C++ (`forward_kernel`, `bce_loss_kernel`, `backward_kernel`, `sgd_update_kernel`).
- **Warp-Level Shuffles (`__shfl_down_sync`)**: Reductions for loss and gradient accumulations occur directly in GPU register space without global memory latency.
- **Python Bridge via Pybind11 / PyTorch C++ Extension**: PyTorch tensors on GPU are passed directly as contiguous memory pointers to the CUDA kernels.
- **Auto JIT-Compilation**: The module automatically compiles Just-In-Time if not pre-built, requiring zero manual build setup.

---

## 🛠️ Requirements

```bash
pip install torch pandas numpy scikit-learn
```

---

## 🚀 How to Run

### 📍 Running on Google Colab

1. Create a new notebook on [Google Colab](https://colab.research.google.com/) and enable GPU:
   - Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ select **T4 GPU** (or A100).
2. Clone or upload your repository, then run in a Colab code cell:
   ```bash
   !git clone <YOUR_REPO_URL>
   %cd <YOUR_REPO_NAME>/01_logistic_regression
   !pip install pandas scikit-learn
   !python train_titanic.py
   ```

### 📍 Running Locally (Linux / Windows with NVIDIA GPU)

1. Navigate to the module folder:
   ```bash
   cd 01_logistic_regression
   ```
2. Run the Titanic training script directly:
   ```bash
   python train_titanic.py
   ```
*(Optional: Ahead-of-Time compilation)*:
```bash
pip install -e .
```

---

## 📊 Titanic Training Output Example

```text
================================================================
       CUDA ML Models: Logistic Regression on Titanic Data      
================================================================

[INFO] Raw Dataset shape: (891, 12)
[INFO] Training samples: 712, Test samples: 179
[CUDA ML] Training Logistic Regression on 712 samples with 9 features...
[CUDA ML] Device: NVIDIA GeForce RTX 3080

  Epoch     1/ 3000 | Loss: 0.69315 | Train Acc: 61.66%
  Epoch   300/ 3000 | Loss: 0.45281 | Train Acc: 80.48%
  Epoch   600/ 3000 | Loss: 0.44312 | Train Acc: 80.76%
  ...
  Epoch  3000/ 3000 | Loss: 0.43849 | Train Acc: 81.32%

----------------------------------------------------------------
                   Test Evaluation Metrics                      
----------------------------------------------------------------
  Test Accuracy:  80.45%
  Precision:      76.56%
  Recall:         71.01%
  F1-Score:       73.68%
  ROC-AUC Score:  0.8624
================================================================
```

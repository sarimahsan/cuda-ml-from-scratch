# 05 - Gated Recurrent Unit (GRU) in Pure CUDA C++

A high-performance **Gated Recurrent Unit (GRU)** recurrent neural network engineered from scratch in **pure CUDA C++ GPU kernel subsystems**, featuring fused in-register 3-gate evaluation (Reset, Update, Candidate), precomputed sequence input GEMMs, exact Backpropagation Through Time (BPTT), PyTorch C++ bindings with custom Autograd integration, and an end-to-end character language model.

---

## 📂 Modular Architecture & Directory Layout

```text
05_gru/
├── csrc/
│   ├── binding.cpp         # PyTorch C++ / Pybind11 registration module
│   ├── gru_cell.cu         # Fused in-register 3-gate forward and backward GPU kernels
│   └── sequence.cu         # High-throughput sequence forward & BPTT backward engine
├── gru.py                  # PyTorch high-level interface (CUDAGRUCell, CUDAGRU, CUDAGRULanguageModel)
├── benchmark.py            # Numerical correctness verification & throughput scaling suite
├── train_sequence.py       # Character-level Language Model training on Shakespeare corpus
├── setup.py                # Setuptools configuration for AOT building
└── README.md               # Technical documentation, mathematical proofs & benchmark guide
```

---

## 📐 Mathematical Formulation (Matching `torch.nn.GRU`)

For a sequence of length $T$, batch size $N$, input dimension $D$, and hidden dimension $H$:

### 1. Forward Pass Equations
For timestep $t \in \{0, 1, \dots, T-1\}$, input $\mathbf{x}_t \in \mathbb{R}^{N \times D}$, and previous state $\mathbf{h}_{t-1} \in \mathbb{R}^{N \times H}$:

1. **Precomputed Input Projections ($1$ GEMM for whole sequence):**
   $$\mathbf{G}_{ih} = \mathbf{X}_{\text{flat}} \mathbf{W}_{ih} + \mathbf{b}_{ih} \quad \in \mathbb{R}^{(T \cdot N) \times 3H}$$
   where $\mathbf{W}_{ih} \in \mathbb{R}^{D \times 3H}$ and $\mathbf{b}_{ih} \in \mathbb{R}^{3H}$.

2. **Recurrent Projections:**
   $$\mathbf{G}_{hh} = \mathbf{h}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_{hh} \quad \in \mathbb{R}^{N \times 3H}$$
   where $\mathbf{W}_{hh} \in \mathbb{R}^{H \times 3H}$ and $\mathbf{b}_{hh} \in \mathbb{R}^{3H}$.

3. **Fused 3-Gate Evaluations in Registers:**
   - **Reset Gate:** $\mathbf{r}_t = \sigma\left(\mathbf{G}_{ih, t}^{[0:H]} + \mathbf{G}_{hh, t}^{[0:H]}\right) = \frac{1}{1 + e^{-\left(\mathbf{G}_{ih, t}^{[0:H]} + \mathbf{G}_{hh, t}^{[0:H]}\right)}}$
   - **Update Gate:** $\mathbf{z}_t = \sigma\left(\mathbf{G}_{ih, t}^{[H:2H]} + \mathbf{G}_{hh, t}^{[H:2H]}\right) = \frac{1}{1 + e^{-\left(\mathbf{G}_{ih, t}^{[H:2H]} + \mathbf{G}_{hh, t}^{[H:2H]}\right)}}$
   - **Candidate Hidden State:** $\mathbf{n}_t = \tanh\left(\mathbf{G}_{ih, t}^{[2H:3H]} + \mathbf{r}_t \odot \mathbf{G}_{hh, t}^{[2H:3H]}\right)$
   - **Hidden State Output:** $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{n}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1}$

---

### 2. Backward Pass Equations (BPTT from $t = T-1 \to 0$)

1. **Total Incoming Gradient:**
   $$\mathbf{dh}_t = \mathbf{dH}_{\text{seq}}[t] + \mathbf{dh}_{\text{next}}$$

2. **Step Gradients in Registers:**
   $$\mathbf{dn}_t = \mathbf{dh}_t \odot (1 - \mathbf{z}_t)$$
   $$\mathbf{dz}_t = \mathbf{dh}_t \odot (\mathbf{h}_{t-1} - \mathbf{n}_t)$$
   $$\mathbf{dz}_{z, t} = \mathbf{dz}_t \odot \mathbf{z}_t \odot (1 - \mathbf{z}_t)$$
   $$\mathbf{dz}_{n, t} = \mathbf{dn}_t \odot (1 - \mathbf{n}_t^2)$$
   $$\mathbf{dr}_t = \mathbf{dz}_{n, t} \odot \mathbf{G}_{hh, t}^{[2H:3H]}$$
   $$\mathbf{dz}_{r, t} = \mathbf{dr}_t \odot \mathbf{r}_t \odot (1 - \mathbf{r}_t)$$
   $$\mathbf{dG}_{hh, n} = \mathbf{dz}_{n, t} \odot \mathbf{r}_t$$

3. **Recurrent Gradient Propagation:**
   $$\mathbf{dG}_{hh, t} = \left[ \mathbf{dz}_{r, t} \,,\, \mathbf{dz}_{z, t} \,,\, \mathbf{dG}_{hh, n} \right] \in \mathbb{R}^{N \times 3H}$$
   $$\mathbf{dh}_{\text{next}} = (\mathbf{dh}_t \odot \mathbf{z}_t) + \mathbf{dG}_{hh, t} \mathbf{W}_{hh}^T$$

4. **Batched Parameter Reductions:**
   $$\mathbf{dG}_{ih} = \left[ \mathbf{dz}_{r} \,,\, \mathbf{dz}_{z} \,,\, \mathbf{dz}_{n} \right] \in \mathbb{R}^{(T \cdot N) \times 3H}$$
   $$\nabla_{\mathbf{W}_{ih}} \mathcal{L} = \mathbf{X}_{\text{flat}}^T \mathbf{dG}_{ih}, \quad \nabla_{\mathbf{b}_{ih}} \mathcal{L} = \sum \mathbf{dG}_{ih}$$
   $$\nabla_{\mathbf{W}_{hh}} \mathcal{L} = \mathbf{H}_{\text{prev\_all}}^T \mathbf{dG}_{hh}, \quad \nabla_{\mathbf{b}_{hh}} \mathcal{L} = \sum \mathbf{dG}_{hh}$$
   $$\nabla_{\mathbf{X}_{\text{seq}}} \mathcal{L} = \mathbf{dG}_{ih} \mathbf{W}_{ih}^T$$

---

## ⚡ Key CUDA Optimizations

1. **Precomputed Flattened Input GEMM:**
   Instead of launching $T$ separate small matrix multiplications of shape $[N, D] \times [D, 3H]$, we reshape the sequence into $[T \cdot N, D]$ and compute the entire input projection in **a single GEMM launch**, eliminating $T-1$ host launch latencies.
2. **Fused 3-Gate In-Register Execution:**
   All 3 gates ($\mathbf{r}_t, \mathbf{z}_t, \mathbf{n}_t$) and state updates are computed in a single kernel pass directly in SM 32-bit registers without intermediate DRAM roundtrips.
3. **Batched BPTT Transposed GEMMs:**
   Parameter gradients across the entire temporal sequence are aggregated in a single transposed GEMM (`gemm_TN`), eliminating per-step atomic reduction contention.

---

## 📊 Empirical Benchmarks (Tesla T4 GPU, 150 Repetitions)

### 1. Isolated Phase Profiling ($T=64, N=64, D=128, H=256$)

| Execution Phase | Custom CUDA (Mean $\pm$ Std) | PyTorch cuDNN (Mean $\pm$ Std) | CUDA Median | cuDNN Median | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Forward Only** | $2.375 \pm 0.584\text{ ms}$ | $1.503 \pm 0.024\text{ ms}$ | $2.203\text{ ms}$ | $1.506\text{ ms}$ | $0.68\times$ |
| **Backward Only** | $3.396 \pm 0.549\text{ ms}$ | $2.314 \pm 0.023\text{ ms}$ | $3.233\text{ ms}$ | $2.314\text{ ms}$ | $0.72\times$ |
| **Forward + Backward (BPTT)** | $5.708 \pm 0.925\text{ ms}$ | $3.981 \pm 0.124\text{ ms}$ | $5.517\text{ ms}$ | $3.988\text{ ms}$ | $0.72\times$ |

### 2. Statistical Distribution & Percentiles (Full BPTT)

| Metric | Custom CUDA GRU | PyTorch Native cuDNN |
| :--- | :---: | :---: |
| **Mean $\pm$ Std Dev** | $5.7082 \pm 0.9249\text{ ms}$ | $3.9809 \pm 0.1243\text{ ms}$ |
| **Median ($P_{50}$)** | **$5.5171\text{ ms}$** | **$3.9883\text{ ms}$** |
| **$P_5$ (5th Percentile)** | $5.1695\text{ ms}$ | $3.8457\text{ ms}$ |
| **$P_{95}$ (95th Percentile)** | $6.9152\text{ ms}$ | $4.1085\text{ ms}$ |
| **Min / Max Range** | $[4.989,\, 10.867]\text{ ms}$ | $[3.361,\, 4.616]\text{ ms}$ |

### 3. Macrobenchmark: Sequence & Dimension Scaling ($150$ Timed Runs)

| Configuration ($T \times N \times D \times H$) | CUDA Median | cuDNN Median | CUDA $P_5 - P_{95}$ Range | CUDA Throughput | Speedup vs cuDNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $T=32, N=32, D=128, H=128$ | $3.295\text{ ms}$ | $1.101\text{ ms}$ | $[3.07, 4.61]\text{ ms}$ | $310,744\text{ tok/s}$ | $0.33\times$ |
| $T=64, N=64, D=128, H=256$ | $5.447\text{ ms}$ | $3.974\text{ ms}$ | $[5.15, 6.54]\text{ ms}$ | $752,026\text{ tok/s}$ | $0.73\times$ |
| **$T=128, N=64, D=256, H=512$** | **$21.928\text{ ms}$** | **$22.414\text{ ms}$** | **$[21.46, 23.02]\text{ ms}$** | **$373,592\text{ tok/s}$** | **$1.02\times$ 🚀** |
| **$T=128, N=128, D=256, H=512$** | **$37.734\text{ ms}$** | **$40.300\text{ ms}$** | **$[36.87, 38.71]\text{ ms}$** | **$434,199\text{ tok/s}$** | **$1.07\times$ 🚀** |
| **$T=256, N=64, D=256, H=512$** | **$45.973\text{ ms}$** | **$47.262\text{ ms}$** | **$[44.83, 47.90]\text{ ms}$** | **$356,386\text{ tok/s}$** | **$1.03\times$ 🚀** |

---

## 🚀 Quickstart & Usage

### 1. Run Verification & Throughput Benchmarks
```bash
python 05_gru/benchmark.py --warmup 50 --reps 150
```

### 2. Train Character-Level Language Model (Shakespeare)
```bash
python 05_gru/train_sequence.py --epochs 10 --seq_len 48 --batch_size 32
```


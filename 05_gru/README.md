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
| **Forward Only** | $1.900 \pm 0.588\text{ ms}$ | $1.520 \pm 0.029\text{ ms}$ | $1.727\text{ ms}$ | $1.522\text{ ms}$ | **$0.88\times$** |
| **Backward Only** | $2.903 \pm 0.219\text{ ms}$ | $2.343 \pm 0.022\text{ ms}$ | $2.859\text{ ms}$ | $2.342\text{ ms}$ | **$0.82\times$** |
| **Forward + Backward (BPTT)** | $4.759 \pm 0.587\text{ ms}$ | $3.971 \pm 0.073\text{ ms}$ | $4.595\text{ ms}$ | $3.965\text{ ms}$ | **$0.86\times$** |

### 2. Statistical Distribution & Percentiles (Full BPTT)

| Metric | Custom CUDA GRU | PyTorch Native cuDNN |
| :--- | :---: | :---: |
| **Mean $\pm$ Std Dev** | $4.7594 \pm 0.5868\text{ ms}$ | $3.9713 \pm 0.0731\text{ ms}$ |
| **Median ($P_{50}$)** | **$4.5946\text{ ms}$** | **$3.9651\text{ ms}$** |
| **$P_5$ (5th Percentile)** | $4.3276\text{ ms}$ | $3.8880\text{ ms}$ |
| **$P_{95}$ (95th Percentile)** | $5.9891\text{ ms}$ | $4.0685\text{ ms}$ |
| **Min / Max Range** | $[4.076,\, 7.965]\text{ ms}$ | $[3.617,\, 4.125]\text{ ms}$ |

### 3. Macrobenchmark: Sequence & Dimension Scaling ($150$ Timed Runs)

| Configuration ($T \times N \times D \times H$) | CUDA Median | cuDNN Median | CUDA $P_5 - P_{95}$ Range | CUDA Throughput | Speedup vs cuDNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $T=32, N=32, D=128, H=128$ | $2.621\text{ ms}$ | $1.100\text{ ms}$ | $[2.48, 2.89]\text{ ms}$ | $390,706\text{ tok/s}$ | $0.42\times$ |
| $T=64, N=64, D=128, H=256$ | $4.231\text{ ms}$ | $4.019\text{ ms}$ | $[4.03, 5.07]\text{ ms}$ | $968,047\text{ tok/s}$ | $0.95\times$ |
| **$T=128, N=64, D=256, H=512$** | **$21.428\text{ ms}$** | **$22.817\text{ ms}$** | **$[20.89, 22.63]\text{ ms}$** | **$382,305\text{ tok/s}$** | **$1.06\times$ 🚀** |
| **$T=128, N=128, D=256, H=512$** | **$37.137\text{ ms}$** | **$40.328\text{ ms}$** | **$[36.61, 38.29]\text{ ms}$** | **$441,174\text{ tok/s}$** | **$1.09\times$ (+9%) 🚀** |
| **$T=256, N=64, D=256, H=512$** | **$45.583\text{ ms}$** | **$48.267\text{ ms}$** | **$[44.59, 47.28]\text{ ms}$** | **$359,433\text{ tok/s}$** | **$1.06\times$ 🚀** |

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


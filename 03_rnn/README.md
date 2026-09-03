# 03 — Basic Elman Recurrent Neural Network (RNN) in Pure CUDA C++

A hardware-accelerated implementation of the **Basic (Elman) Recurrent Neural Network (RNN)** written from scratch in CUDA C++ and bound to PyTorch.

---

## 1. Mathematical Formulation

For an input sequence $X \in \mathbb{R}^{T \times N \times D}$, hidden state $h_t \in \mathbb{R}^{N \times H}$, input weights $W_{ih} \in \mathbb{R}^{D \times H}$, recurrent weights $W_{hh} \in \mathbb{R}^{H \times H}$, and biases $b_{ih}, b_{hh} \in \mathbb{R}^{H}$:

### Forward Pass
1. **Precomputed Input GEMM:**
   $$G_{ih} = X_{\text{flat}} \cdot W_{ih} + b_{ih} \quad \in \mathbb{R}^{(T \cdot N) \times H}$$

2. **Recurrent Temporal Loop ($t = 0 \to T-1$):**
   $$z_t = G_{ih}[t] + h_{t-1} \cdot W_{hh} + b_{hh}$$
   $$h_t = \tanh(z_t) \quad (\text{or } \operatorname{ReLU}(z_t))$$

### Backward Pass (Backpropagation Through Time — BPTT)
For $t = T-1 \to 0$:
1. **Accumulate Recurrent Gradient:**
   $$dh_t = dH_{\text{seq}}[t] + dh_{\text{next}}$$

2. **Step Activation Gradient:**
   $$dz_t = dh_t \odot (1 - h_t^2) \quad (\text{for } \tanh)$$

3. **Recurrent Gradient Propagation:**
   $$dh_{\text{next}} = dz_t \cdot W_{hh}^T$$

4. **Batched Parameter Reductions:**
   $$dW_{ih} = X_{\text{all}}^T \cdot dZ_{\text{all}}, \quad dW_{hh} = H_{\text{prev\_all}}^T \cdot dZ_{\text{all}}$$
   $$db_{ih} = db_{hh} = \sum_{n, t} dZ_{\text{all}}, \quad dX_{\text{seq}} = dZ_{\text{all}} \cdot W_{ih}^T$$

---

## 2. CUDA Architecture Highlights

- **Precomputed Input Projections:** Flattens the entire sequence $[T \cdot N, D]$ into a single call to the centralized `gemm_register_tiled` engine, eliminating $T-1$ separate kernel launches.
- **Fused Recurrence Kernel:** Merges bias addition and $\tanh(z)$ in-register per timestep.
- **Batched BPTT Reductions:** Reduces parameter gradients across all timesteps simultaneously with hardware-tuned transposed GEMMs (`gemm_TN`, `gemm_NT`).

---

## 3. Directory Structure

```
03_rnn/
├── csrc/
│   ├── binding.cpp         # PyTorch C++ / Pybind11 interface
│   ├── rnn_cell.cu         # In-place accumulation and fused step kernels
│   └── sequence.cu         # High-speed sequence forward and BPTT backward loops
├── rnn.py                  # High-level Python interface (CUDARNNCell, CUDARNN, CUDARNNLanguageModel)
├── benchmark.py            # Microbenchmarks & correctness suite vs torch.nn.RNN
├── train_char_lm.py        # Character language model training demo
├── setup.py                # Standalone JIT setup script
└── README.md               # Documentation
```

---

## 4. Benchmarking & Verification

Run the benchmark suite:
```bash
python 03_rnn/benchmark.py
```

Run character language model training:
```bash
python 03_rnn/train_char_lm.py --epochs 10 --seq_len 32 --batch_size 16
```

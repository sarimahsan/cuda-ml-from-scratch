import os
import math
import time
from typing import List, Optional, Union
import torch
from torch.utils.cpp_extension import load

# -------------------------------------------------------------------------
# Dynamic Extension Loader (Compiles modular CUDA sources)
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "linear.cu"),
    os.path.join(current_dir, "csrc", "activations.cu"),
    os.path.join(current_dir, "csrc", "softmax_loss.cu"),
    os.path.join(current_dir, "csrc", "optimizers.cu"),
]

try:
    import cuda_mlp as _ext  # type: ignore
except ImportError:
    print("[INFO] Compiling modular CUDA MLP extension via JIT...")
    _ext = load(
        name="cuda_mlp",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )


class CUDAMLP:
    """
    High-Performance Multi-Layer Perceptron (MLP) built from modular CUDA C++ kernels.
    Modules:
      - csrc/linear.cu: 2D Tiled Shared-Memory GEMMs for linear transformations.
      - csrc/activations.cu: Forward and backward activations (ReLU, GELU, Sigmoid).
      - csrc/softmax_loss.cu: Numerically stable Softmax and Cross-Entropy loss with reductions.
      - csrc/optimizers.cu: Vectorized GPU optimizers (SGD with Momentum, Adam).
    """

    def __init__(
        self,
        layer_sizes: List[int] = [784, 128, 10],
        activation: str = "relu",
        lr: float = 1e-3,
        optimizer: str = "adam",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        device: str = "cuda",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please run on a GPU-enabled machine/Colab.")

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation.lower()
        self.lr = lr
        self.optimizer_type = optimizer.lower()
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.step_count = 0

        self.weights: List[torch.Tensor] = []
        self.biases: List[torch.Tensor] = []

        # Optimizer state buffers
        self.v_w: List[torch.Tensor] = []
        self.v_b: List[torch.Tensor] = []
        self.m_w: List[torch.Tensor] = []
        self.m_b: List[torch.Tensor] = []

        self.loss_history: List[float] = []
        self._init_parameters()

    def _init_parameters(self, seed: int = 42):
        torch.manual_seed(seed)
        self.weights = []
        self.biases = []
        self.v_w = []
        self.v_b = []
        self.m_w = []
        self.m_b = []

        for i in range(self.num_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            # He (Kaiming) Normal Initialization: std = sqrt(2 / fan_in)
            std = math.sqrt(2.0 / fan_in)
            W = torch.randn(fan_in, fan_out, device=self.device, dtype=torch.float32) * std
            b = torch.zeros(fan_out, device=self.device, dtype=torch.float32)

            self.weights.append(W.contiguous())
            self.biases.append(b.contiguous())

            # Optimizer state buffers
            self.v_w.append(torch.zeros_like(W))
            self.v_b.append(torch.zeros_like(b))
            self.m_w.append(torch.zeros_like(W))
            self.m_b.append(torch.zeros_like(b))

    def _to_tensor(self, data, dtype=torch.float32) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=dtype).contiguous()
        return torch.tensor(data, device=self.device, dtype=dtype).contiguous()

    def _apply_activation_forward(self, Z: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return _ext.gelu_forward(Z)
        elif self.activation == "sigmoid":
            return _ext.sigmoid_forward(Z)
        else:
            return _ext.relu_forward(Z)

    def _apply_activation_backward(self, dA: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return _ext.gelu_backward(dA, Z)
        elif self.activation == "sigmoid":
            return _ext.sigmoid_backward(dA, Z)
        else:
            return _ext.relu_backward(dA, Z)

    def forward_pass(self, X: torch.Tensor):
        """
        Full forward pass through all MLP layers using modular CUDA kernels.
        Returns:
            activations (list of A), pre_activations (list of Z)
        """
        A = X
        activations = [A]
        pre_activations = []

        for i in range(self.num_layers):
            W = self.weights[i]
            b = self.biases[i]

            # 1. Tiled CUDA Linear Forward (csrc/linear.cu)
            Z = _ext.linear_forward(A, W, b)
            pre_activations.append(Z)

            if i < self.num_layers - 1:
                # Hidden layer: Activation (csrc/activations.cu)
                A = self._apply_activation_forward(Z)
                activations.append(A)
            else:
                # Output logits
                activations.append(Z)

        return activations, pre_activations

    def fit(
        self,
        X,
        y,
        epochs: int = 20,
        batch_size: int = 128,
        validation_data: Optional[tuple] = None,
        verbose: bool = True,
    ):
        """
        Train the MLP using modular CUDA kernels across batches.
        """
        X_cuda = self._to_tensor(X, dtype=torch.float32)
        y_cuda = self._to_tensor(y, dtype=torch.int64).view(-1)

        N = X_cuda.size(0)
        num_batches = (N + batch_size - 1) // batch_size

        if verbose:
            print("=" * 65)
            print(f"   CUDA ML: Multi-Layer Perceptron (Architecture: {self.layer_sizes})")
            print("=" * 65)
            print(f"[CUDA ML] Training {N} samples on {torch.cuda.get_device_name(0)}")
            print(f"[CUDA ML] Batch size: {batch_size} | Optimizer: {self.optimizer_type.upper()} | Act: {self.activation.upper()}")
            print("-" * 65)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            t0 = time.perf_counter()

            # Random permutation of training indices per epoch
            perm = torch.randperm(N, device=self.device)
            X_shuffled = X_cuda[perm]
            y_shuffled = y_cuda[perm]

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, N)

                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                # 1. Forward Pass
                activations, pre_activations = self.forward_pass(batch_X)
                logits = activations[-1]

                # 2. Softmax + Cross-Entropy Loss & Output Gradient (csrc/softmax_loss.cu)
                probs, loss_t, dZ = _ext.softmax_cross_entropy(logits, batch_y)
                loss_val = loss_t.item()
                epoch_loss += loss_val * (end_idx - start_idx)

                # 3. Backpropagation through all layers
                cur_dZ = dZ
                grads_W = [None] * self.num_layers
                grads_b = [None] * self.num_layers

                for layer_idx in reversed(range(self.num_layers)):
                    A_in = activations[layer_idx]
                    W = self.weights[layer_idx]
                    compute_dX = layer_idx > 0

                    # Linear backward (csrc/linear.cu): dW = A_in^T * cur_dZ, db = sum(cur_dZ), dX = cur_dZ * W^T
                    dW, db, dX = _ext.linear_backward(cur_dZ, A_in, W, compute_dX)
                    grads_W[layer_idx] = dW
                    grads_b[layer_idx] = db

                    if compute_dX:
                        # Activation backward (csrc/activations.cu): dZ_prev = dX * act'(Z_prev)
                        Z_prev = pre_activations[layer_idx - 1]
                        cur_dZ = self._apply_activation_backward(dX, Z_prev)

                # 4. Optimizer Step (csrc/optimizers.cu)
                self.step_count += 1
                for layer_idx in range(self.num_layers):
                    if self.optimizer_type == "adam":
                        _ext.adam_step(
                            self.weights[layer_idx],
                            self.m_w[layer_idx],
                            self.v_w[layer_idx],
                            grads_W[layer_idx],
                            self.lr,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            self.step_count,
                        )
                        _ext.adam_step(
                            self.biases[layer_idx],
                            self.m_b[layer_idx],
                            self.v_b[layer_idx],
                            grads_b[layer_idx],
                            self.lr,
                            self.beta1,
                            self.beta2,
                            self.eps,
                            self.step_count,
                        )
                    else:
                        # SGD with Momentum
                        _ext.sgd_momentum_step(
                            self.weights[layer_idx],
                            self.v_w[layer_idx],
                            grads_W[layer_idx],
                            self.lr,
                            self.momentum,
                        )
                        _ext.sgd_momentum_step(
                            self.biases[layer_idx],
                            self.v_b[layer_idx],
                            grads_b[layer_idx],
                            self.lr,
                            self.momentum,
                        )

            epoch_time = (time.perf_counter() - t0) * 1000.0  # ms
            avg_loss = epoch_loss / N
            self.loss_history.append(avg_loss)

            if verbose:
                train_acc = self.score(batch_X, batch_y)
                val_str = ""
                if validation_data is not None:
                    val_acc = self.score(validation_data[0], validation_data[1])
                    val_str = f" | Val Acc: {val_acc:6.2f}%"

                print(
                    f"  Epoch {epoch:3d}/{epochs:3d} | CE Loss: {avg_loss:.4f} | "
                    f"Train Acc: {train_acc:6.2f}%{val_str} | Time: {epoch_time:6.2f} ms"
                )

        return self

    def predict_proba(self, X) -> torch.Tensor:
        X_cuda = self._to_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            activations, _ = self.forward_pass(X_cuda)
            logits = activations[-1]
            probs, _, _ = _ext.softmax_cross_entropy(
                logits, torch.zeros(logits.size(0), dtype=torch.int64, device=self.device)
            )
        return probs

    def predict(self, X) -> torch.Tensor:
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_cuda = self._to_tensor(y, dtype=torch.int64).view(-1)
        return (preds == y_cuda).float().mean().item() * 100.0

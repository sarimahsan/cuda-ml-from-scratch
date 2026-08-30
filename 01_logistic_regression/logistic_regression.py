import os
import shutil
import subprocess
import sys
import torch
from torch.utils.cpp_extension import load

# -------------------------------------------------------------------------
# Dynamic Extension Loader (JIT compile or load installed AOT extension)
# -------------------------------------------------------------------------
def _ensure_ninja():
    if shutil.which("ninja") is None:
        try:
            import ninja  # type: ignore
        except ImportError:
            print("[INFO] 'ninja' build tool not detected. Auto-installing ninja for JIT compilation...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "--quiet"])
            except Exception as e:
                print(f"[WARNING] Auto-installing ninja failed: {e}")

_ensure_ninja()

current_dir = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(current_dir, "csrc", "logistic_regression_kernel.cu"),
]

try:
    import cuda_logistic_regression as _ext  # type: ignore
except ImportError:
    print("[INFO] Compiling CUDA Logistic Regression extension via JIT...")
    _ext = load(
        name="cuda_logistic_regression",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )


class CUDALogisticRegression:
    """
    High-Performance Binary Logistic Regression using pure custom CUDA C++ kernels.
    Supports NumPy arrays and PyTorch Tensors on GPU.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 1000, device: str = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please run on a GPU-enabled machine/Colab.")

        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.w = None
        self.b = None
        self.loss_history = []

    def _to_tensor(self, data, dtype=torch.float32) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=dtype).contiguous()
        else:
            return torch.tensor(data, device=self.device, dtype=dtype).contiguous()

    def fit(self, X, y, batch_size: int = None, verbose: bool = True):
        """
        Train the model using custom CUDA forward, loss, backward, and SGD kernels.
        """
        X_cuda = self._to_tensor(X)
        y_cuda = self._to_tensor(y).view(-1)

        N, D = X_cuda.shape

        # Initialize weights and bias on GPU
        self.w = torch.zeros(D, device=self.device, dtype=torch.float32)
        self.b = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.loss_history = []

        if verbose:
            print(f"[CUDA ML] Training Logistic Regression on {N} samples with {D} features...")
            print(f"[CUDA ML] Device: {torch.cuda.get_device_name(0)}")

        for epoch in range(1, self.epochs + 1):
            # 1. Forward Pass (CUDA Kernel)
            y_hat = _ext.forward(X_cuda, self.w, self.b)

            # 2. Compute Loss (CUDA Reduction Kernel)
            loss = _ext.bce_loss(y_hat, y_cuda).item()
            self.loss_history.append(loss)

            # 3. Backward Pass (CUDA Feature-Reduction Gradients)
            grad_w, grad_b = _ext.backward(X_cuda, y_hat, y_cuda)

            # 4. Optimizer Step (CUDA In-place Update Kernel)
            _ext.sgd_step(self.w, self.b, grad_w, grad_b, self.lr)

            if verbose and (epoch % (self.epochs // 10 or 1) == 0 or epoch == 1):
                preds = (y_hat >= 0.5).float()
                acc = (preds == y_cuda).float().mean().item() * 100.0
                print(f"  Epoch {epoch:5d}/{self.epochs:5d} | Loss: {loss:.5f} | Train Acc: {acc:.2f}%")

        return self

    def predict_proba(self, X) -> torch.Tensor:
        """
        Compute predicted probabilities using CUDA forward kernel.
        """
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before predicting.")

        X_cuda = self._to_tensor(X)
        with torch.no_grad():
            y_hat = _ext.forward(X_cuda, self.w, self.b)
        return y_hat

    def predict(self, X, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute binary classification predictions {0, 1}.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).to(torch.int32)

    def score(self, X, y) -> float:
        """
        Calculate classification accuracy on test dataset.
        """
        preds = self.predict(X)
        y_cuda = self._to_tensor(y).to(torch.int32).view(-1)
        return (preds == y_cuda).float().mean().item() * 100.0

    @property
    def weights(self):
        return self.w.cpu().numpy() if self.w is not None else None

    @property
    def bias(self):
        return self.b.cpu().item() if self.b is not None else None

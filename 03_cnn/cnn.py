import os
import math
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# -------------------------------------------------------------------------
# Dynamic Extension Loader (Compiles modular CUDA sources)
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
root_dir = os.path.dirname(current_dir)
kernels_dir = os.path.join(root_dir, "kernels")
common_dir = os.path.join(root_dir, "00_common", "include")

sources = [
    os.path.join(current_dir, "csrc", "binding.cpp"),
    os.path.join(kernels_dir, "src", "convolution.cu"),
    os.path.join(kernels_dir, "src", "pooling.cu"),
    os.path.join(kernels_dir, "src", "gemm.cu"),
    os.path.join(kernels_dir, "src", "activation.cu"),
    os.path.join(kernels_dir, "src", "softmax.cu"),
    os.path.join(kernels_dir, "src", "reduction.cu"),
    os.path.join(kernels_dir, "src", "elementwise.cu"),
    os.path.join(kernels_dir, "src", "optimizers.cu"),
]

try:
    import cuda_cnn as _ext  # type: ignore
except ImportError:
    print("[INFO] Compiling modular CUDA CNN extension via JIT using centralized kernels...")
    _ext = load(
        name="cuda_cnn",
        sources=sources,
        extra_include_paths=[os.path.join(kernels_dir, "include"), common_dir],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )



# -------------------------------------------------------------------------
# Autograd Function Wrappers for Modular CUDA Operations
# -------------------------------------------------------------------------
class CUDAConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor], stride: int = 1, pad: int = 0):
        ctx.save_for_backward(X, W)
        ctx.stride = stride
        ctx.pad = pad
        ctx.has_bias = b is not None
        return _ext.conv2d_forward(X, W, b if b is not None else torch.tensor([], device=X.device), stride, pad)

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        X, W = ctx.saved_tensors
        dW, db, dX = _ext.conv2d_backward(dO.contiguous(), X, W, ctx.stride, ctx.pad, ctx.needs_input_grad[0])
        db_grad = db if ctx.has_bias else None
        return dX, dW, db_grad, None, None


class CUDAMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, pool_h: int = 2, pool_w: int = 2, stride: int = 2, pad: int = 0):
        P, mask = _ext.maxpool2d_forward(X, pool_h, pool_w, stride, pad)
        ctx.save_for_backward(mask)
        ctx.input_shape = X.shape
        return P

    @staticmethod
    def backward(ctx, dP: torch.Tensor):
        mask, = ctx.saved_tensors
        N, C, H_in, W_in = ctx.input_shape
        dX = _ext.maxpool2d_backward(dP.contiguous(), mask, N, C, H_in, W_in)
        return dX, None, None, None, None


class CUDALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor]):
        ctx.save_for_backward(X, W)
        ctx.has_bias = b is not None
        return _ext.linear_forward(X, W, b if b is not None else torch.tensor([], device=X.device))

    @staticmethod
    def backward(ctx, dZ: torch.Tensor):
        X, W = ctx.saved_tensors
        dW, db, dX = _ext.linear_backward(dZ.contiguous(), X, W, ctx.needs_input_grad[0])
        db_grad = db if ctx.has_bias else None
        return dX, dW, db_grad


class CUDAReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z: torch.Tensor):
        ctx.save_for_backward(Z)
        return _ext.relu_forward(Z)

    @staticmethod
    def backward(ctx, dA: torch.Tensor):
        Z, = ctx.saved_tensors
        return _ext.relu_backward(dA.contiguous(), Z)


class CUDASoftmaxCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor):
        loss, probs, dZ = _ext.softmax_cross_entropy(logits, targets)
        ctx.save_for_backward(dZ)
        ctx.probs = probs
        return loss, probs

    @staticmethod
    def backward(ctx, grad_loss, grad_probs=None):
        dZ, = ctx.saved_tensors
        if grad_loss is not None:
            return dZ * grad_loss, None
        return dZ, None


# -------------------------------------------------------------------------
# Custom Module Layers
# -------------------------------------------------------------------------
class CUDAConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True, device: str = "cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming He Normal initialization
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device) * std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return CUDAConv2dFunction.apply(X, self.weight, self.bias, self.stride, self.padding)


class CUDAMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return CUDAMaxPool2dFunction.apply(X, self.kernel_size, self.kernel_size, self.stride, self.padding)


class CUDALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: str = "cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device=device) * std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return CUDALinearFunction.apply(X, self.weight, self.bias)


class CUDAReLU(nn.Module):
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return CUDAReLUFunction.apply(Z)


# -------------------------------------------------------------------------
# CUDACNN Model Coordinator
# -------------------------------------------------------------------------
class CUDACNN(nn.Module):
    """
    Modular Convolutional Neural Network (CNN) built on custom CUDA C++ kernels.
    Architecture:
      - Conv1 (3x3, pad 1) -> ReLU -> MaxPool (2x2)
      - Conv2 (3x3, pad 1) -> ReLU -> MaxPool (2x2)
      - Flatten -> Linear -> ReLU -> Linear -> Softmax Cross-Entropy
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_height: int = 28,
        in_width: int = 28,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_hidden: int = 128,
        num_classes: int = 10,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.step_count = 0
        self.device = device

        # Conv1: [N, in_channels, 28, 28] -> [N, conv1_channels, 28, 28]
        self.conv1 = CUDAConv2d(in_channels, conv1_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.relu1 = CUDAReLU()
        self.pool1 = CUDAMaxPool2d(kernel_size=2, stride=2)  # -> [N, conv1_channels, 14, 14]

        # Conv2: [N, conv1_channels, 14, 14] -> [N, conv2_channels, 14, 14]
        self.conv2 = CUDAConv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.relu2 = CUDAReLU()
        self.pool2 = CUDAMaxPool2d(kernel_size=2, stride=2)  # -> [N, conv2_channels, 7, 7]

        # Spatial output size calculation
        h_pool2 = in_height // 4
        w_pool2 = in_width // 4
        self.flat_dim = conv2_channels * h_pool2 * w_pool2

        # Fully Connected Layers
        self.fc1 = CUDALinear(self.flat_dim, fc_hidden, device=device)
        self.relu3 = CUDAReLU()
        self.fc2 = CUDALinear(fc_hidden, num_classes, device=device)

        # Optimizer State Buffers
        self.v_params = {}
        self.m_params = {}
        for name, p in self.named_parameters():
            self.v_params[name] = torch.zeros_like(p)
            self.m_params[name] = torch.zeros_like(p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class logits.
        """
        out = self.pool1(self.relu1(self.conv1(X)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.reshape(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        logits = self.fc2(out)
        return logits

    def compute_loss_and_probs(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Numerically stable CUDA Softmax Cross-Entropy with full autograd support.
        targets: One-hot encoded [N, num_classes] or class indices [N].
        """
        if targets.dim() == 1 or targets.size(1) != self.num_classes:
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=self.num_classes).float()
        
        loss, probs = CUDASoftmaxCrossEntropyFunction.apply(logits, targets)
        return loss, probs, None

    def step_optimizer(self, method: str = "momentum", beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        In-place GPU parameter updates without host synchronizations.
        """
        self.step_count += 1
        for name, p in self.named_parameters():
            if p.grad is not None:
                if method == "momentum":
                    _ext.sgd_momentum_step(p.data, self.v_params[name], p.grad.data, self.lr, self.momentum)
                elif method == "adam":
                    _ext.adam_step(
                        p.data, self.m_params[name], self.v_params[name],
                        p.grad.data, self.lr, beta1, beta2, eps, self.step_count
                    )
                p.grad.zero_()

    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """
        Evaluate classification accuracy and average loss across a dataset.
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.forward(X)
            targets_onehot = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes).float()
            loss, probs = CUDASoftmaxCrossEntropyFunction.apply(logits, targets_onehot)

            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

        self.train()
        return total_loss / total, correct / total

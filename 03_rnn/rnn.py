"""
=============================================================================
Modular Pure CUDA Basic RNN: Python High-Level Interface
=============================================================================
Provides high-level PyTorch-compatible abstractions:
  - CUDARNNCell: Single timestep recurrent cell
  - CUDARNN: Multi-timestep sequence model with full BPTT autograd
  - CUDARNNLanguageModel: Character/token language model with text generation
=============================================================================
"""

import os
import math
import shutil
import subprocess
import sys
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


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
    os.path.join(current_dir, "csrc", "rnn_cell.cu"),
    os.path.join(current_dir, "csrc", "sequence.cu"),
    os.path.join(kernels_dir, "src", "gemm.cu"),
    os.path.join(kernels_dir, "src", "softmax.cu"),
    os.path.join(kernels_dir, "src", "reduction.cu"),
    os.path.join(kernels_dir, "src", "elementwise.cu"),
    os.path.join(kernels_dir, "src", "optimizers.cu"),
]

try:
    import cuda_rnn as _ext  # type: ignore
except ImportError:
    print("[INFO] Compiling modular CUDA RNN extension via JIT using centralized kernels...")
    _ext = load(
        name="cuda_rnn",
        sources=sources,
        extra_include_paths=[
            os.path.join(kernels_dir, "include"),
            common_dir,
            os.path.join(current_dir, "csrc"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )
    print("[INFO] CUDA RNN JIT compilation successful!")


# -----------------------------------------------------------------------------
# 1. Custom Autograd Function for High-Performance Sequence RNN
# -----------------------------------------------------------------------------
class _CUDARNNSequenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_seq, W_ih, b_ih, W_hh, b_hh, h_0, act_type):
        ctx.act_type = act_type
        # Call high-throughput sequence forward
        H_seq, h_T, _ = _ext.rnn_forward_sequence(
            X_seq.contiguous(),
            W_ih.contiguous(),
            b_ih.contiguous() if b_ih is not None else torch.empty(0, device=X_seq.device),
            W_hh.contiguous(),
            b_hh.contiguous() if b_hh is not None else torch.empty(0, device=X_seq.device),
            h_0.contiguous(),
            act_type
        )
        ctx.save_for_backward(X_seq, W_ih, W_hh, h_0, H_seq)
        return H_seq, h_T

    @staticmethod
    def backward(ctx, dH_seq, dh_T):
        X_seq, W_ih, W_hh, h_0, H_seq = ctx.saved_tensors
        act_type = ctx.act_type

        # If gradient is provided for h_T, accumulate into dH_seq[-1]
        dH_seq_total = dH_seq.clone() if dH_seq is not None else torch.zeros_like(H_seq)
        if dh_T is not None:
            dH_seq_total[-1] += dh_T

        dW_ih, db_ih, dW_hh, db_hh, dX_seq = _ext.rnn_backward_sequence(
            dH_seq_total.contiguous(),
            X_seq.contiguous(),
            W_ih.contiguous(),
            W_hh.contiguous(),
            h_0.contiguous(),
            H_seq.contiguous(),
            act_type
        )

        db_ih_ret = db_ih if ctx.needs_input_grad[2] else None
        db_hh_ret = db_hh if ctx.needs_input_grad[4] else None

        return dX_seq, dW_ih, db_ih_ret, dW_hh, db_hh_ret, None, None


# -----------------------------------------------------------------------------
# 2. PyTorch High-Level Modules
# -----------------------------------------------------------------------------
class CUDARNNCell(nn.Module):
    """
    Single-step Basic Elman RNN Cell:
        h_t = act(x_t * W_ih + b_ih + h_{t-1} * W_hh + b_hh)
    """
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh", bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity.lower()
        self.act_type = 0 if self.nonlinearity == "tanh" else 1

        # Parameter weights (matching standard PyTorch shapes)
        self.weight_ih = nn.Parameter(torch.empty(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)
        if self.bias_ih is not None:
            nn.init.uniform_(self.bias_ih, -std, std)
            nn.init.uniform_(self.bias_hh, -std, std)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        b_ih = self.bias_ih if self.bias_ih is not None else torch.empty(0, device=x.device)
        b_hh = self.bias_hh if self.bias_hh is not None else torch.empty(0, device=x.device)
        return _ext.rnn_cell_forward(x, h_prev, self.weight_ih, b_ih, self.weight_hh, b_hh, self.act_type)


class CUDARNN(nn.Module):
    """
    Multi-timestep Basic Elman RNN Sequence Model with hardware-accelerated BPTT.
    """
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh", bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity.lower()
        self.act_type = 0 if self.nonlinearity == "tanh" else 1

        self.weight_ih = nn.Parameter(torch.empty(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)
        if self.bias_ih is not None:
            nn.init.uniform_(self.bias_ih, -std, std)
            nn.init.uniform_(self.bias_hh, -std, std)

    def forward(self, x_seq: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: [T, N, D] sequence tensor
            h_0: [N, H] initial hidden state (optional)
        Returns:
            H_seq: [T, N, H] hidden states across all timesteps
            h_T: [N, H] final hidden state at step T
        """
        T, N, D = x_seq.shape
        if h_0 is None:
            h_0 = torch.zeros(N, self.hidden_size, device=x_seq.device, dtype=x_seq.dtype)
        return _CUDARNNSequenceFunction.apply(
            x_seq, self.weight_ih, self.bias_ih, self.weight_hh, self.bias_hh, h_0, self.act_type
        )


class CUDARNNLanguageModel(nn.Module):
    """
    Complete Character/Token Language Model using pure CUDA primitives.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, nonlinearity: str = "tanh"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = CUDARNN(embed_dim, hidden_size, nonlinearity=nonlinearity)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [T, N]
        embeddings = self.embed(tokens) # [T, N, D]
        H_seq, h_T = self.rnn(embeddings, h_0) # [T, N, H]
        logits = self.head(H_seq) # [T, N, V]
        return logits, h_T

    @torch.no_grad()
    def generate(self, seed_indices: list, max_length: int = 100, temperature: float = 0.8) -> list:
        self.eval()
        device = next(self.parameters()).device
        generated = list(seed_indices)
        cur_token = torch.tensor([[seed_indices[-1]]], device=device, dtype=torch.long)
        h = None

        # Prime with seed
        if len(seed_indices) > 1:
            prime_tokens = torch.tensor(seed_indices[:-1], device=device, dtype=torch.long).unsqueeze(1)
            _, h = self.forward(prime_tokens)

        for _ in range(max_length):
            emb = self.embed(cur_token) # [1, 1, D]
            H_seq, h = self.rnn(emb, h)
            logits = self.head(H_seq.squeeze(0)) / max(temperature, 1e-5) # [1, V]
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_idx)
            cur_token = torch.tensor([[next_idx]], device=device, dtype=torch.long)

        return generated

"""
=============================================================================
Modular Pure CUDA LSTM: Python High-Level Interface
=============================================================================
Provides high-level PyTorch-compatible abstractions:
  - CUDALSTMCell: Single timestep recurrent cell (Modular or Fused mode)
  - CUDALSTM: Multi-timestep sequence model with full BPTT
  - CUDALSTMLanguageModel: Character/token language model with generation
=============================================================================
"""

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
    os.path.join(current_dir, "csrc", "input_gate.cu"),
    os.path.join(current_dir, "csrc", "forget_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_candidate_gate.cu"),
    os.path.join(current_dir, "csrc", "output_gate.cu"),
    os.path.join(current_dir, "csrc", "cell_state.cu"),
    os.path.join(current_dir, "csrc", "fused_gates.cu"),
    os.path.join(current_dir, "csrc", "linear.cu"),
    os.path.join(current_dir, "csrc", "softmax_loss.cu"),
    os.path.join(current_dir, "csrc", "optimizers.cu"),
]

try:
    import cuda_lstm as _ext  # type: ignore
except ImportError:
    if torch.cuda.is_available():
        print("[INFO] Compiling modular CUDA LSTM extension via JIT...")
        _ext = load(
            name="cuda_lstm",
            sources=sources,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    else:
        _ext = None


class CUDALSTMCell:
    """
    Single Timestep LSTM Recurrent Cell powered by pure CUDA kernels.
    Can be executed in modular separate-gate mode or fused mode.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mode: str = "fused",
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mode = mode.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        std = 1.0 / math.sqrt(hidden_dim)
        # Weight matrices: W_ih [input_dim, 4 * hidden_dim], W_hh [hidden_dim, 4 * hidden_dim]
        self.W_ih = (torch.randn(input_dim, 4 * hidden_dim, device=self.device) * std).contiguous()
        self.b_ih = torch.zeros(4 * hidden_dim, device=self.device).contiguous()
        self.W_hh = (torch.randn(hidden_dim, 4 * hidden_dim, device=self.device) * std).contiguous()
        self.b_hh = torch.zeros(4 * hidden_dim, device=self.device).contiguous()

        # Initialize forget gate bias to 1.0 for better gradient retention
        with torch.no_grad():
            self.b_ih[hidden_dim : 2 * hidden_dim] = 1.0

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward step at single timestep t:
        Returns:
            h_next [N x H], c_next [N x H]
        """
        if _ext is None:
            raise RuntimeError("CUDA extension is not loaded or CUDA device unavailable.")

        # 1. Projections: G_ih = x_t * W_ih, G_hh = h_prev * W_hh
        G_ih = _ext.linear_forward(x_t, self.W_ih, self.b_ih)
        G_hh = _ext.linear_forward(h_prev, self.W_hh, self.b_hh)
        G_total = G_ih + G_hh

        if self.mode == "modular":
            # Modular mode: compute each gate separately via dedicated kernels
            H = self.hidden_dim
            Z_i = G_total[:, :H].contiguous()
            Z_f = G_total[:, H : 2 * H].contiguous()
            Z_g = G_total[:, 2 * H : 3 * H].contiguous()
            Z_o = G_total[:, 3 * H : 4 * H].contiguous()

            i_val = _ext.input_gate_forward(Z_i)
            f_val = _ext.forget_gate_forward(Z_f)
            g_val = _ext.candidate_gate_forward(Z_g)
            o_val = _ext.output_gate_forward(Z_o)

            c_next, tanh_c, h_next = _ext.cell_state_forward(f_val, c_prev, i_val, g_val, o_val)
            return h_next, c_next
        else:
            # Fused mode: single kernel pass
            h_next, c_next, _, _ = _ext.fused_lstm_gates_forward(G_total, c_prev)
            return h_next, c_next


class CUDALSTM:
    """
    Multi-Timestep Sequence LSTM with complete Backpropagation Through Time (BPTT).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lr: float = 1e-3,
        clip_norm: float = 1.0,
        mode: str = "fused",
        device: str = "cuda",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_norm = clip_norm
        self.mode = mode.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.step_count = 0

        std = 1.0 / math.sqrt(hidden_dim)
        self.W_ih = (torch.randn(input_dim, 4 * hidden_dim, device=self.device) * std).contiguous()
        self.b_ih = torch.zeros(4 * hidden_dim, device=self.device).contiguous()
        self.W_hh = (torch.randn(hidden_dim, 4 * hidden_dim, device=self.device) * std).contiguous()
        self.b_hh = torch.zeros(4 * hidden_dim, device=self.device).contiguous()

        with torch.no_grad():
            self.b_ih[hidden_dim : 2 * hidden_dim] = 1.0

        # Optimizer buffers (Adam)
        self.m_W_ih = torch.zeros_like(self.W_ih)
        self.v_W_ih = torch.zeros_like(self.W_ih)
        self.m_b_ih = torch.zeros_like(self.b_ih)
        self.v_b_ih = torch.zeros_like(self.b_ih)

        self.m_W_hh = torch.zeros_like(self.W_hh)
        self.v_W_hh = torch.zeros_like(self.W_hh)
        self.m_b_hh = torch.zeros_like(self.b_hh)
        self.v_b_hh = torch.zeros_like(self.b_hh)

    def forward_sequence(
        self,
        X_seq: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        c_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        Forward pass over sequence X_seq [T, N, D].
        Returns:
            H_seq [T, N, H], (h_T, c_T), cache dictionary for BPTT
        """
        T, N, D = X_seq.shape
        H = self.hidden_dim

        if h_0 is None:
            h_0 = torch.zeros(N, H, device=self.device)
        if c_0 is None:
            c_0 = torch.zeros(N, H, device=self.device)

        H_list = [h_0]
        C_list = [c_0]
        G_total_list = []
        G_act_list = []
        Tanh_C_list = []

        cur_h = h_0
        cur_c = c_0

        for t in range(T):
            x_t = X_seq[t].contiguous()
            G_ih = _ext.linear_forward(x_t, self.W_ih, self.b_ih)
            G_hh = _ext.linear_forward(cur_h, self.W_hh, self.b_hh)
            G_tot = G_ih + G_hh
            G_total_list.append(G_tot)

            if self.mode == "modular":
                Z_i = G_tot[:, :H].contiguous()
                Z_f = G_tot[:, H : 2 * H].contiguous()
                Z_g = G_tot[:, 2 * H : 3 * H].contiguous()
                Z_o = G_tot[:, 3 * H : 4 * H].contiguous()

                i_val = _ext.input_gate_forward(Z_i)
                f_val = _ext.forget_gate_forward(Z_f)
                g_val = _ext.candidate_gate_forward(Z_g)
                o_val = _ext.output_gate_forward(Z_o)

                next_c, tanh_c, next_h = _ext.cell_state_forward(f_val, cur_c, i_val, g_val, o_val)
                g_act = torch.cat([i_val, f_val, g_val, o_val], dim=1)
            else:
                next_h, next_c, g_act, tanh_c = _ext.fused_lstm_gates_forward(G_tot, cur_c)

            H_list.append(next_h)
            C_list.append(next_c)
            G_act_list.append(g_act)
            Tanh_C_list.append(tanh_c)

            cur_h = next_h
            cur_c = next_c

        out_H = torch.stack(H_list[1:], dim=0)
        cache = {
            "X_seq": X_seq,
            "H_list": H_list,
            "C_list": C_list,
            "G_total_list": G_total_list,
            "G_act_list": G_act_list,
            "Tanh_C_list": Tanh_C_list,
        }
        return out_H, (cur_h, cur_c), cache

    def backward_sequence(
        self,
        dH_seq: torch.Tensor,
        cache: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact BPTT backward pass.
        Returns:
            dW_ih, db_ih, dW_hh, db_hh, dX_seq
        """
        X_seq = cache["X_seq"]
        H_list = cache["H_list"]
        C_list = cache["C_list"]
        G_act_list = cache["G_act_list"]
        Tanh_C_list = cache["Tanh_C_list"]

        T, N, D = X_seq.shape
        H = self.hidden_dim

        dW_ih = torch.zeros_like(self.W_ih)
        db_ih = torch.zeros_like(self.b_ih)
        dW_hh = torch.zeros_like(self.W_hh)
        db_hh = torch.zeros_like(self.b_hh)
        dX_seq = torch.zeros_like(X_seq)

        dh_next = torch.zeros(N, H, device=self.device)
        dc_next = torch.zeros(N, H, device=self.device)

        for t in reversed(range(T)):
            x_t = X_seq[t]
            h_prev = H_list[t]
            c_prev = C_list[t]
            c_curr = C_list[t + 1]
            tanh_c = Tanh_C_list[t]
            g_act = G_act_list[t]

            dh_curr = dH_seq[t] + dh_next

            if self.mode == "modular":
                i_val = g_act[:, :H]
                f_val = g_act[:, H : 2 * H]
                g_val = g_act[:, 2 * H : 3 * H]
                o_val = g_act[:, 3 * H : 4 * H]

                dc_prev, df, di, dg, do_t = _ext.cell_state_backward(
                    dh_curr, dc_next, o_val, tanh_c, c_prev, f_val, i_val, g_val
                )

                dZ_i = _ext.input_gate_backward(di, i_val)
                dZ_f = _ext.forget_gate_backward(df, f_val)
                dZ_g = _ext.candidate_gate_backward(dg, g_val)
                dZ_o = _ext.output_gate_backward(do_t, o_val)

                dG_tot = torch.cat([dZ_i, dZ_f, dZ_g, dZ_o], dim=1)
                dc_next = dc_prev
            else:
                dG_tot, dc_next = _ext.fused_lstm_gates_backward(
                    dh_curr, dc_next, g_act, c_prev, c_curr, tanh_c
                )

            # dW_ih += x_t^T * dG_tot, db_ih += sum(dG_tot)
            dW_ih_t, db_ih_t, dX_t = _ext.linear_backward(dG_tot, x_t, self.W_ih, True)
            dW_ih += dW_ih_t
            db_ih += db_ih_t
            dX_seq[t] = dX_t

            # dW_hh += h_prev^T * dG_tot, db_hh += sum(dG_tot), dh_prev = dG_tot * W_hh^T
            dW_hh_t, db_hh_t, dh_next = _ext.linear_backward(dG_tot, h_prev, self.W_hh, True)
            dW_hh += dW_hh_t
            db_hh += db_hh_t

        return dW_ih, db_ih, dW_hh, db_hh, dX_seq

    def step(self, grads: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Apply gradient clipping and Adam optimizer updates in-place.
        """
        dW_ih, db_ih, dW_hh, db_hh = grads
        self.step_count += 1

        # In-place GPU gradient norm clipping
        _ext.clip_grad_norm([dW_ih, db_ih, dW_hh, db_hh], self.clip_norm)

        # In-place Adam updates
        _ext.adam_step(self.W_ih, self.m_W_ih, self.v_W_ih, dW_ih, self.lr, 0.9, 0.999, 1e-8, self.step_count)
        _ext.adam_step(self.b_ih, self.m_b_ih, self.v_b_ih, db_ih, self.lr, 0.9, 0.999, 1e-8, self.step_count)
        _ext.adam_step(self.W_hh, self.m_W_hh, self.v_W_hh, dW_hh, self.lr, 0.9, 0.999, 1e-8, self.step_count)
        _ext.adam_step(self.b_hh, self.m_b_hh, self.v_b_hh, db_hh, self.lr, 0.9, 0.999, 1e-8, self.step_count)


class CUDALSTMLanguageModel:
    """
    End-to-End Sequence Language Model (e.g. Shakespeare Character LM).
    Integrates LSTM + Linear Classification Head + Softmax Cross-Entropy Loss.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        lr: float = 2e-3,
        clip_norm: float = 1.0,
        mode: str = "fused",
        device: str = "cuda",
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_norm = clip_norm
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # LSTM backbone
        self.lstm = CUDALSTM(
            input_dim=vocab_size,
            hidden_dim=hidden_dim,
            lr=lr,
            clip_norm=clip_norm,
            mode=mode,
            device=device,
        )

        # Linear projection head: H -> Vocab
        std_head = 1.0 / math.sqrt(hidden_dim)
        self.W_out = (torch.randn(hidden_dim, vocab_size, device=self.device) * std_head).contiguous()
        self.b_out = torch.zeros(vocab_size, device=self.device).contiguous()

        self.m_W_out = torch.zeros_like(self.W_out)
        self.v_W_out = torch.zeros_like(self.W_out)
        self.m_b_out = torch.zeros_like(self.b_out)
        self.v_b_out = torch.zeros_like(self.b_out)
        self.step_count = 0

    def train_step(self, x_seq_indices: torch.Tensor, y_seq_targets: torch.Tensor) -> float:
        """
        Performs one full Forward + BPTT + Adam optimization step.
        x_seq_indices: [T, N] integer token indices
        y_seq_targets: [T, N] integer target token indices
        """
        T, N = x_seq_indices.shape
        # One-hot encoding on GPU: [T, N, Vocab]
        X_seq = torch.nn.functional.one_hot(x_seq_indices, num_classes=self.vocab_size).float().contiguous()

        # 1. LSTM forward sequence pass
        H_seq, (h_last, c_last), cache = self.lstm.forward_sequence(X_seq)

        # 2. Linear projection head: [T*N, H] * [H, V] -> [T*N, V]
        H_flat = H_seq.view(T * N, self.hidden_dim).contiguous()
        logits_flat = _ext.linear_forward(H_flat, self.W_out, self.b_out)

        # 3. Sequence Softmax Cross-Entropy Loss & Analytical Logit Gradient
        targets_flat = y_seq_targets.view(T * N).contiguous()
        probs_flat, loss_tensor, dlogits_flat = _ext.softmax_cross_entropy(logits_flat, targets_flat)

        # 4. Backward through Linear Head
        dW_out, db_out, dH_flat = _ext.linear_backward(dlogits_flat, H_flat, self.W_out, True)
        dH_seq = dH_flat.view(T, N, self.hidden_dim).contiguous()

        # 5. Backward through LSTM (BPTT)
        dW_ih, db_ih, dW_hh, db_hh, _ = self.lstm.backward_sequence(dH_seq, cache)

        # 6. Optimization Step
        self.step_count += 1
        _ext.clip_grad_norm([dW_out, db_out], self.clip_norm)
        _ext.adam_step(self.W_out, self.m_W_out, self.v_W_out, dW_out, self.lr, 0.9, 0.999, 1e-8, self.step_count)
        _ext.adam_step(self.b_out, self.m_b_out, self.v_b_out, db_out, self.lr, 0.9, 0.999, 1e-8, self.step_count)

        self.lstm.step((dW_ih, db_ih, dW_hh, db_hh))
        return loss_tensor.item()

    def generate(
        self,
        seed_indices: List[int],
        length: int = 100,
        temperature: float = 0.8,
    ) -> List[int]:
        """
        Autoregressively generates token sequence.
        """
        generated = list(seed_indices)
        h = torch.zeros(1, self.hidden_dim, device=self.device)
        c = torch.zeros(1, self.hidden_dim, device=self.device)

        # Warmup hidden states on seed tokens
        for idx in seed_indices:
            x_t = torch.zeros(1, self.vocab_size, device=self.device)
            x_t[0, idx] = 1.0
            G_ih = _ext.linear_forward(x_t, self.lstm.W_ih, self.lstm.b_ih)
            G_hh = _ext.linear_forward(h, self.lstm.W_hh, self.lstm.b_hh)
            h, c, _, _ = _ext.fused_lstm_gates_forward(G_ih + G_hh, c)

        cur_idx = seed_indices[-1]
        for _ in range(length):
            x_t = torch.zeros(1, self.vocab_size, device=self.device)
            x_t[0, cur_idx] = 1.0
            G_ih = _ext.linear_forward(x_t, self.lstm.W_ih, self.lstm.b_ih)
            G_hh = _ext.linear_forward(h, self.lstm.W_hh, self.lstm.b_hh)
            h, c, _, _ = _ext.fused_lstm_gates_forward(G_ih + G_hh, c)

            logits = _ext.linear_forward(h, self.W_out, self.b_out) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            cur_idx = next_token

        return generated

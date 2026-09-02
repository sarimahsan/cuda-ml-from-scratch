"""
=============================================================================
High-Resolution CUDA LSTM Backward (BPTT) Profiler & Root-Cause Investigator
=============================================================================
Isolates every component of the ~4.09 ms backward execution:
  1. Recurrent Temporal Loop (64 timesteps)
     ├── dW_hh recurrent weight GEMM (H_prev^T * dG_t)
     ├── dH_prev recurrent data GEMM (dG_t * W_hh^T)
     ├── db_hh bias reduction (sum(dG_t))
     ├── Fused Gate Backward kernel (dG_t, dc_prev)
     └── dh accumulation (dh_curr = dH[t] + dh_next)
  2. Batched Input Backward
     ├── dW_ih GEMM (X_all^T * dG_all)
     ├── dX_seq GEMM (dG_all * W_ih^T)
     └── db_ih reduction (sum(dG_all))
  3. Memory Allocation & Buffer Overhead
=============================================================================
"""

import argparse
import time
import torch
import torch.nn as nn
from lstm import CUDALSTM, _ext


def profile_backward_deep_dive(
    seq_len=64,
    batch_size=64,
    input_dim=128,
    hidden_dim=256,
    num_warmup=10,
    num_runs=50,
    device="cuda",
):
    print("\n" + "=" * 85)
    print(" 🔍 DEEP-DIVE PROFILER: INVESTIGATING THE ~4ms BACKWARD LATENCY")
    print("=" * 85)
    print(f"• Sequence Length (T) : {seq_len}")
    print(f"• Batch Size (N)      : {batch_size}")
    print(f"• Input Dimension (D) : {input_dim}")
    print(f"• Hidden Dimension (H): {hidden_dim}")
    print(f"• Profiling Runs      : {num_runs} iterations (after {num_warmup} warmups)")
    print(f"• Device              : {torch.cuda.get_device_name(0)}")
    print("-" * 85)

    if not torch.cuda.is_available() or _ext is None:
        print("[ERROR] CUDA extension not available.")
        return

    # Setup model and dummy inputs
    model = CUDALSTM(input_dim, hidden_dim, mode="fused", device=device)
    X = torch.randn(seq_len, batch_size, input_dim, device=device)
    grad_out = torch.randn(seq_len, batch_size, hidden_dim, device=device)
    h_0 = torch.zeros(batch_size, hidden_dim, device=device)
    c_0 = torch.zeros(batch_size, hidden_dim, device=device)

    # Warmup
    for _ in range(num_warmup):
        out, (h, c), cache = model.forward_sequence(X, h_0, c_0)
        model.backward_sequence(grad_out, cache)
    torch.cuda.synchronize()

    # Forward once to populate cache
    out, (h, c), cache = model.forward_sequence(X, h_0, c_0)

    # -------------------------------------------------------------------------
    # TEST 1: Measure Overall Backward End-to-End
    # -------------------------------------------------------------------------
    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(num_runs):
        model.backward_sequence(grad_out, cache)
    stop_ev.record()
    torch.cuda.synchronize()
    total_bwd_ms = start_ev.elapsed_time(stop_ev) / num_runs

    # -------------------------------------------------------------------------
    # TEST 2: Component-Level Timing Breakdown using CUDA Events
    # -------------------------------------------------------------------------
    H_seq = cache["H_seq"]
    C_seq = cache["C_seq"]
    G_act_seq = cache["G_act_seq"]
    Tanh_C_seq = cache["Tanh_C_seq"]
    four_H = 4 * hidden_dim

    # Accumulators (in ms)
    t_dh_accum = 0.0
    t_gate_bwd = 0.0
    t_gemm_w_hh = 0.0
    t_bias_hh = 0.0
    t_gemm_h_prev = 0.0

    t_gemm_w_ih = 0.0
    t_gemm_x = 0.0
    t_bias_ih = 0.0
    t_alloc = 0.0

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_stop = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        # 1. Allocation timing
        ev_start.record()
        dW_hh = torch.zeros((hidden_dim, four_H), device=device)
        db_hh = torch.zeros((four_H,), device=device)
        dG_all = torch.empty((seq_len * batch_size, four_H), device=device)
        dh_next = torch.zeros((batch_size, hidden_dim), device=device)
        dc_next = torch.zeros((batch_size, hidden_dim), device=device)
        ev_stop.record()
        torch.cuda.synchronize()
        t_alloc += ev_start.elapsed_time(ev_stop)

        # 2. Recurrent loop micro-benchmarks
        for t in reversed(range(seq_len)):
            d_dh_t = grad_out[t]
            h_prev = H_seq[t - 1] if t > 0 else h_0
            d_g_act_t = G_act_seq[t]
            d_c_prev = C_seq[t]
            d_c_curr = C_seq[t + 1]
            d_tanh_c = Tanh_C_seq[t]
            dg_t = dG_all.narrow(0, t * batch_size, batch_size)

            # a) dh accumulation
            ev_start.record()
            dh_next.add_(d_dh_t)
            ev_stop.record()
            torch.cuda.synchronize()
            t_dh_accum += ev_start.elapsed_time(ev_stop)

            # b) Fused gate backward kernel
            ev_start.record()
            _ext.fused_lstm_gates_backward(
                dh_next, dc_next, d_g_act_t, d_c_prev, d_c_curr, d_tanh_c
            )
            ev_stop.record()
            torch.cuda.synchronize()
            t_gate_bwd += ev_start.elapsed_time(ev_stop)

            # c) dW_hh accumulation GEMM: [H x N] x [N x 4H]
            ev_start.record()
            dW_hh.addmm_(h_prev.t(), dg_t)
            ev_stop.record()
            torch.cuda.synchronize()
            t_gemm_w_hh += ev_start.elapsed_time(ev_stop)

            # d) db_hh bias reduction
            ev_start.record()
            db_hh.add_(dg_t.sum(0))
            ev_stop.record()
            torch.cuda.synchronize()
            t_bias_hh += ev_start.elapsed_time(ev_stop)

            # e) dh_prev GEMM: [N x 4H] x [4H x H]
            ev_start.record()
            dh_next = torch.mm(dg_t, model.W_hh.t())
            ev_stop.record()
            torch.cuda.synchronize()
            t_gemm_h_prev += ev_start.elapsed_time(ev_stop)

        # 3. Batched Input Backward
        X_flat = X.reshape(seq_len * batch_size, input_dim)

        ev_start.record()
        dW_ih = torch.mm(X_flat.t(), dG_all)
        ev_stop.record()
        torch.cuda.synchronize()
        t_gemm_w_ih += ev_start.elapsed_time(ev_stop)

        ev_start.record()
        db_ih = dG_all.sum(0)
        ev_stop.record()
        torch.cuda.synchronize()
        t_bias_ih += ev_start.elapsed_time(ev_stop)

        ev_start.record()
        dX_seq = torch.mm(dG_all, model.W_ih.t()).view(seq_len, batch_size, input_dim)
        ev_stop.record()
        torch.cuda.synchronize()
        t_gemm_x += ev_start.elapsed_time(ev_stop)

    # Average times
    t_dh_accum /= num_runs
    t_gate_bwd /= num_runs
    t_gemm_w_hh /= num_runs
    t_bias_hh /= num_runs
    t_gemm_h_prev /= num_runs

    recurrent_total = t_dh_accum + t_gate_bwd + t_gemm_w_hh + t_bias_hh + t_gemm_h_prev

    t_gemm_w_ih /= num_runs
    t_bias_ih /= num_runs
    t_gemm_x /= num_runs
    input_total = t_gemm_w_ih + t_bias_ih + t_gemm_x

    t_alloc /= num_runs
    accounted_ms = recurrent_total + input_total + t_alloc
    dispatch_overhead = max(0.0, total_bwd_ms - accounted_ms)

    print(f"\n{'Subsystem / Micro-Operation':<46} | {'Time (ms)':>10} | {'% of Backward':>14} | {'Calls / Sequence':>16}")
    print("=" * 95)
    print(f"▶ TOTAL MEASURED BACKWARD PASS                     | {total_bwd_ms:9.3f} ms | {'100.0%':>14} | {'1 call':>16}")
    print("-" * 95)

    print(f"1. RECURRENT BPTT LOOP (T={seq_len} steps)              | {recurrent_total:9.3f} ms | {recurrent_total/total_bwd_ms*100:13.1f}% | {seq_len*5:14} calls")
    print(f"   ├── dW_hh Recurrent GEMM (H_prev^T * dG)        | {t_gemm_w_hh:9.3f} ms | {t_gemm_w_hh/total_bwd_ms*100:13.1f}% | {seq_len:14} calls")
    print(f"   ├── dH_prev Data GEMM (dG * W_hh^T)             | {t_gemm_h_prev:9.3f} ms | {t_gemm_h_prev/total_bwd_ms*100:13.1f}% | {seq_len:14} calls")
    print(f"   ├── db_hh Bias Reduction (sum(dG))              | {t_bias_hh:9.3f} ms | {t_bias_hh/total_bwd_ms*100:13.1f}% | {seq_len:14} calls")
    print(f"   ├── Fused Gate Backward Kernel (dG, dc_prev)    | {t_gate_bwd:9.3f} ms | {t_gate_bwd/total_bwd_ms*100:13.1f}% | {seq_len:14} calls")
    print(f"   └── dh Accumulation Kernel (dH[t] + dh_next)    | {t_dh_accum:9.3f} ms | {t_dh_accum/total_bwd_ms*100:13.1f}% | {seq_len:14} calls")
    print("-" * 95)

    print(f"2. BATCHED INPUT BACKWARD                          | {input_total:9.3f} ms | {input_total/total_bwd_ms*100:13.1f}% | {'3 calls':>16}")
    print(f"   ├── dW_ih GEMM (X_all^T * dG_all)               | {t_gemm_w_ih:9.3f} ms | {t_gemm_w_ih/total_bwd_ms*100:13.1f}% | {'1 call':>16}")
    print(f"   ├── dX_seq Data GEMM (dG_all * W_ih^T)          | {t_gemm_x:9.3f} ms | {t_gemm_x/total_bwd_ms*100:13.1f}% | {'1 call':>16}")
    print(f"   └── db_ih Bias Reduction (sum(dG_all))          | {t_bias_ih:9.3f} ms | {t_bias_ih/total_bwd_ms*100:13.1f}% | {'1 call':>16}")
    print("-" * 95)

    print(f"3. Buffer Allocation & Graph Housekeeping          | {t_alloc:9.3f} ms | {t_alloc/total_bwd_ms*100:13.1f}% | {'5 allocs':>16}")
    print(f"4. Host CPU Queue & Driver Dispatch Overhead       | {dispatch_overhead:9.3f} ms | {dispatch_overhead/total_bwd_ms*100:13.1f}% | {'320 dispatches':>16}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep-dive Profiler for CUDA LSTM Backward Pass")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length (default: 64)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--input_dim", type=int, default=128, help="Input dim (default: 128)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dim (default: 256)")
    parser.add_argument("--runs", type=int, default=50, help="Number of profiling runs (default: 50)")
    args = parser.parse_args()

    profile_backward_deep_dive(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_runs=args.runs,
    )

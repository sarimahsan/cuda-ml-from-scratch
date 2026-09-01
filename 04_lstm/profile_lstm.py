"""
=============================================================================
Modular Pure CUDA LSTM: Microsecond Fine-Grained Component Profiler
=============================================================================
Isolates and profiles EVERY individual sub-operation within Forward and
Backward passes:
  Recurrent Backward:
    ├── dW_hh GEMM (H_prev^T * dG)
    ├── db_hh reduction (sum(dG))
    ├── dH_prev GEMM (dG * W_hh^T)
    └── accumulation / memory overhead
  Input Backward:
    ├── dW_ih GEMM (X_t^T * dG)
    ├── db_ih reduction (sum(dG))
    ├── dX GEMM (dG * W_ih^T)
    └── accumulation / memory overhead
  Fused Gate Backward (dG, dc_prev)
=============================================================================
"""

import argparse
import time
import torch
from lstm import CUDALSTM, _ext


def profile_lstm_breakdown(
    seq_len=64,
    batch_size=64,
    input_dim=128,
    hidden_dim=256,
    num_warmup=10,
    num_runs=50,
    device="cuda",
):
    print("=" * 85)
    print(" 🔬 PURE CUDA LSTM: FINE-GRAINED HARDWARE KERNEL PROFILER")
    print("=" * 85)
    print(f"• Sequence Length (T) : {seq_len}")
    print(f"• Batch Size (N)      : {batch_size}")
    print(f"• Input Dimension (D) : {input_dim}")
    print(f"• Hidden Dimension (H): {hidden_dim}")
    print(f"• Profiling Runs      : {num_runs} iterations (after {num_warmup} warmups)")
    print(f"• Device              : {torch.cuda.get_device_name(0)}")
    print("-" * 85)

    model = CUDALSTM(input_dim, hidden_dim, mode="fused", device=device)
    X_seq = torch.randn(seq_len, batch_size, input_dim, device=device)
    grad_out = torch.randn(seq_len, batch_size, hidden_dim, device=device)
    h_0 = torch.zeros(batch_size, hidden_dim, device=device)
    c_0 = torch.zeros(batch_size, hidden_dim, device=device)

    # Forward CUDA Events
    fwd_gemm_ih_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    fwd_gemm_hh_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    fwd_add_ev     = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    fwd_gate_ev    = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]

    # Backward CUDA Events (Fine-grained per-kernel isolation)
    bwd_gate_ev    = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]

    # Recurrent Backward Events
    bwd_dw_hh_ev   = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_db_hh_ev   = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_dh_prev_ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_acc_hh_ev  = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]

    # Input Backward Events
    bwd_dw_ih_ev   = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_db_ih_ev   = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_dx_ev      = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]
    bwd_acc_ih_ev  = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(seq_len)]

    # Warmup
    for _ in range(num_warmup):
        out, (h_n, c_n), cache = model.forward_sequence(X_seq, h_0, c_0)
        model.backward_sequence(grad_out, cache)
    torch.cuda.synchronize()

    # Forward Accumulators
    t_fwd_gemm_ih = 0.0
    t_fwd_gemm_hh = 0.0
    t_fwd_add = 0.0
    t_fwd_gate = 0.0
    t_total_fwd = 0.0

    # Backward Accumulators
    t_bwd_gate = 0.0
    t_bwd_dw_hh = 0.0
    t_bwd_db_hh = 0.0
    t_bwd_dh_prev = 0.0
    t_bwd_acc_hh = 0.0

    t_bwd_dw_ih = 0.0
    t_bwd_db_ih = 0.0
    t_bwd_dx = 0.0
    t_bwd_acc_ih = 0.0
    t_total_bwd = 0.0

    start_total_ev = torch.cuda.Event(enable_timing=True)
    stop_total_ev = torch.cuda.Event(enable_timing=True)

    for run in range(num_runs):
        # ---------------------------------------------------------------------
        # 1. PROFILE FORWARD PASS
        # ---------------------------------------------------------------------
        start_total_ev.record()

        H_list = [h_0]
        C_list = [c_0]
        G_act_list = []
        Tanh_C_list = []

        cur_h = h_0
        cur_c = c_0

        for t in range(seq_len):
            x_t = X_seq[t].contiguous()

            # 1.1 Input GEMM + Bias
            fwd_gemm_ih_ev[t][0].record()
            G_ih = _ext.linear_forward(x_t, model.W_ih, model.b_ih)
            fwd_gemm_ih_ev[t][1].record()

            # 1.2 Recurrent GEMM + Bias
            fwd_gemm_hh_ev[t][0].record()
            G_hh = _ext.linear_forward(cur_h, model.W_hh, model.b_hh)
            fwd_gemm_hh_ev[t][1].record()

            # 1.3 Pre-activation sum
            fwd_add_ev[t][0].record()
            G_tot = G_ih + G_hh
            fwd_add_ev[t][1].record()

            # 1.4 Fused Gate Forward
            fwd_gate_ev[t][0].record()
            next_h, next_c, g_act, tanh_c = _ext.fused_lstm_gates_forward(G_tot, cur_c)
            fwd_gate_ev[t][1].record()

            H_list.append(next_h)
            C_list.append(next_c)
            G_act_list.append(g_act)
            Tanh_C_list.append(tanh_c)

            cur_h = next_h
            cur_c = next_c

        stop_total_ev.record()
        torch.cuda.synchronize()
        t_total_fwd += start_total_ev.elapsed_time(stop_total_ev)

        for t in range(seq_len):
            t_fwd_gemm_ih += fwd_gemm_ih_ev[t][0].elapsed_time(fwd_gemm_ih_ev[t][1])
            t_fwd_gemm_hh += fwd_gemm_hh_ev[t][0].elapsed_time(fwd_gemm_hh_ev[t][1])
            t_fwd_add += fwd_add_ev[t][0].elapsed_time(fwd_add_ev[t][1])
            t_fwd_gate += fwd_gate_ev[t][0].elapsed_time(fwd_gate_ev[t][1])

        # ---------------------------------------------------------------------
        # 2. PROFILE BACKWARD PASS
        # ---------------------------------------------------------------------
        start_total_ev.record()

        dW_ih = torch.zeros_like(model.W_ih)
        db_ih = torch.zeros_like(model.b_ih)
        dW_hh = torch.zeros_like(model.W_hh)
        db_hh = torch.zeros_like(model.b_hh)
        dX_seq = torch.zeros_like(X_seq)

        dh_next = torch.zeros(batch_size, hidden_dim, device=device)
        dc_next = torch.zeros(batch_size, hidden_dim, device=device)

        for t in reversed(range(seq_len)):
            x_t = X_seq[t]
            h_prev = H_list[t]
            c_prev = C_list[t]
            c_curr = C_list[t + 1]
            tanh_c = Tanh_C_list[t]
            g_act = G_act_list[t]

            dh_curr = grad_out[t] + dh_next

            # 2.1 Fused Gate Backward
            bwd_gate_ev[t][0].record()
            dG_tot, dc_next = _ext.fused_lstm_gates_backward(
                dh_curr, dc_next, g_act, c_prev, c_curr, tanh_c
            )
            bwd_gate_ev[t][1].record()

            # 2.2 RECURRENT BACKWARD BREAKDOWN
            # a) dW_hh GEMM: H_prev^T * dG
            bwd_dw_hh_ev[t][0].record()
            dW_hh_t = _ext.gemm_backward_weights(h_prev, dG_tot)
            bwd_dw_hh_ev[t][1].record()

            # b) db_hh reduction: sum(dG)
            bwd_db_hh_ev[t][0].record()
            db_hh_t = _ext.gemm_backward_bias(dG_tot)
            bwd_db_hh_ev[t][1].record()

            # c) dH_prev GEMM: dG * W_hh^T
            bwd_dh_prev_ev[t][0].record()
            dh_next = _ext.gemm_backward_data(dG_tot, model.W_hh)
            bwd_dh_prev_ev[t][1].record()

            # d) Accumulation / Memory Ops
            bwd_acc_hh_ev[t][0].record()
            dW_hh += dW_hh_t
            db_hh += db_hh_t
            bwd_acc_hh_ev[t][1].record()

            # 2.3 INPUT BACKWARD BREAKDOWN
            # a) dW_ih GEMM: X_t^T * dG
            bwd_dw_ih_ev[t][0].record()
            dW_ih_t = _ext.gemm_backward_weights(x_t, dG_tot)
            bwd_dw_ih_ev[t][1].record()

            # b) db_ih reduction: sum(dG)
            bwd_db_ih_ev[t][0].record()
            db_ih_t = _ext.gemm_backward_bias(dG_tot)
            bwd_db_ih_ev[t][1].record()

            # c) dX GEMM: dG * W_ih^T
            bwd_dx_ev[t][0].record()
            dX_t = _ext.gemm_backward_data(dG_tot, model.W_ih)
            dX_seq[t] = dX_t
            bwd_dx_ev[t][1].record()

            # d) Accumulation / Memory Ops
            bwd_acc_ih_ev[t][0].record()
            dW_ih += dW_ih_t
            db_ih += db_ih_t
            bwd_acc_ih_ev[t][1].record()

        stop_total_ev.record()
        torch.cuda.synchronize()
        t_total_bwd += start_total_ev.elapsed_time(stop_total_ev)

        for t in range(seq_len):
            t_bwd_gate += bwd_gate_ev[t][0].elapsed_time(bwd_gate_ev[t][1])

            t_bwd_dw_hh += bwd_dw_hh_ev[t][0].elapsed_time(bwd_dw_hh_ev[t][1])
            t_bwd_db_hh += bwd_db_hh_ev[t][0].elapsed_time(bwd_db_hh_ev[t][1])
            t_bwd_dh_prev += bwd_dh_prev_ev[t][0].elapsed_time(bwd_dh_prev_ev[t][1])
            t_bwd_acc_hh += bwd_acc_hh_ev[t][0].elapsed_time(bwd_acc_hh_ev[t][1])

            t_bwd_dw_ih += bwd_dw_ih_ev[t][0].elapsed_time(bwd_dw_ih_ev[t][1])
            t_bwd_db_ih += bwd_db_ih_ev[t][0].elapsed_time(bwd_db_ih_ev[t][1])
            t_bwd_dx += bwd_dx_ev[t][0].elapsed_time(bwd_dx_ev[t][1])
            t_bwd_acc_ih += bwd_acc_ih_ev[t][0].elapsed_time(bwd_acc_ih_ev[t][1])

    # Averages
    fwd_gemm_ih = t_fwd_gemm_ih / num_runs
    fwd_gemm_hh = t_fwd_gemm_hh / num_runs
    fwd_add = t_fwd_add / num_runs
    fwd_gate = t_fwd_gate / num_runs
    total_fwd = t_total_fwd / num_runs

    bwd_gate = t_bwd_gate / num_runs
    bwd_dw_hh = t_bwd_dw_hh / num_runs
    bwd_db_hh = t_bwd_db_hh / num_runs
    bwd_dh_prev = t_bwd_dh_prev / num_runs
    bwd_acc_hh = t_bwd_acc_hh / num_runs
    recurrent_bwd_total = bwd_dw_hh + bwd_db_hh + bwd_dh_prev + bwd_acc_hh

    bwd_dw_ih = t_bwd_dw_ih / num_runs
    bwd_db_ih = t_bwd_db_ih / num_runs
    bwd_dx = t_bwd_dx / num_runs
    bwd_acc_ih = t_bwd_acc_ih / num_runs
    input_bwd_total = bwd_dw_ih + bwd_db_ih + bwd_dx + bwd_acc_ih

    total_bwd = t_total_bwd / num_runs

    fwd_kernels = fwd_gemm_ih + fwd_gemm_hh + fwd_add + fwd_gate
    fwd_overhead = max(0.0, total_fwd - fwd_kernels)

    bwd_kernels = bwd_gate + recurrent_bwd_total + input_bwd_total
    bwd_overhead = max(0.0, total_bwd - bwd_kernels)

    grand_total = total_fwd + total_bwd

    print("\n" + "=" * 85)
    print(" 📊 DETAILED BREAKDOWN OF WHERE THE TIME IS SPENT")
    print("=" * 85)
    print(f"{'Subsystem / Micro-Operation':<45} | {'Time (ms)':>10} | {'% of Phase':>11} | {'% of Total':>11}")
    print("-" * 85)

    # 1. FORWARD PASS
    print(f"▶ FORWARD PASS (Total = {total_fwd:.3f} ms)")
    print(f"  • Input GEMM (X * W_ih + b_ih)              | {fwd_gemm_ih:>9.3f} ms | {fwd_gemm_ih/total_fwd*100:>10.1f}% | {fwd_gemm_ih/grand_total*100:>10.1f}%")
    print(f"  • Recurrent GEMM (H_prev * W_hh + b_hh)     | {fwd_gemm_hh:>9.3f} ms | {fwd_gemm_hh/total_fwd*100:>10.1f}% | {fwd_gemm_hh/grand_total*100:>10.1f}%")
    print(f"  • Pre-activation Sum (G_ih + G_hh)           | {fwd_add:>9.3f} ms | {fwd_add/total_fwd*100:>10.1f}% | {fwd_add/grand_total*100:>10.1f}%")
    print(f"  • Fused Gate Forward Kernel (i,f,g,o,c,h)   | {fwd_gate:>9.3f} ms | {fwd_gate/total_fwd*100:>10.1f}% | {fwd_gate/grand_total*100:>10.1f}%")
    print(f"  • Launch / Python Loop Overhead             | {fwd_overhead:>9.3f} ms | {fwd_overhead/total_fwd*100:>10.1f}% | {fwd_overhead/grand_total*100:>10.1f}%")
    print("-" * 85)

    # 2. BACKWARD PASS
    print(f"▶ BACKWARD PASS (BPTT) (Total = {total_bwd:.3f} ms)")
    print(f"  • Fused Gate Backward Kernel (dG, dc_prev)   | {bwd_gate:>9.3f} ms | {bwd_gate/total_bwd*100:>10.1f}% | {bwd_gate/grand_total*100:>10.1f}%")
    print(f"  • [RECURRENT BACKWARD SUB-TOTAL]            | {recurrent_bwd_total:>9.3f} ms | {recurrent_bwd_total/total_bwd*100:>10.1f}% | {recurrent_bwd_total/grand_total*100:>10.1f}%")
    print(f"    ├── dW_hh GEMM (H_prev^T * dG)            | {bwd_dw_hh:>9.3f} ms | {bwd_dw_hh/total_bwd*100:>10.1f}% | {bwd_dw_hh/grand_total*100:>10.1f}%")
    print(f"    ├── db_hh reduction (sum(dG))             | {bwd_db_hh:>9.3f} ms | {bwd_db_hh/total_bwd*100:>10.1f}% | {bwd_db_hh/grand_total*100:>10.1f}%")
    print(f"    ├── dH_prev GEMM (dG * W_hh^T)            | {bwd_dh_prev:>9.3f} ms | {bwd_dh_prev/total_bwd*100:>10.1f}% | {bwd_dh_prev/grand_total*100:>10.1f}%")
    print(f"    └── Accumulation & Memory Ops (+=)        | {bwd_acc_hh:>9.3f} ms | {bwd_acc_hh/total_bwd*100:>10.1f}% | {bwd_acc_hh/grand_total*100:>10.1f}%")
    print(f"  • [INPUT BACKWARD SUB-TOTAL]                | {input_bwd_total:>9.3f} ms | {input_bwd_total/total_bwd*100:>10.1f}% | {input_bwd_total/grand_total*100:>10.1f}%")
    print(f"    ├── dW_ih GEMM (X_t^T * dG)               | {bwd_dw_ih:>9.3f} ms | {bwd_dw_ih/total_bwd*100:>10.1f}% | {bwd_dw_ih/grand_total*100:>10.1f}%")
    print(f"    ├── db_ih reduction (sum(dG))             | {bwd_db_ih:>9.3f} ms | {bwd_db_ih/total_bwd*100:>10.1f}% | {bwd_db_ih/grand_total*100:>10.1f}%")
    print(f"    ├── dX Data Gradient GEMM (dG * W_ih^T)   | {bwd_dx:>9.3f} ms | {bwd_dx/total_bwd*100:>10.1f}% | {bwd_dx/grand_total*100:>10.1f}%")
    print(f"    └── Accumulation & Memory Ops (+=)        | {bwd_acc_ih:>9.3f} ms | {bwd_acc_ih/total_bwd*100:>10.1f}% | {bwd_acc_ih/grand_total*100:>10.1f}%")
    print(f"  • Launch / Python Loop Overhead             | {bwd_overhead:>9.3f} ms | {bwd_overhead/total_bwd*100:>10.1f}% | {bwd_overhead/grand_total*100:>10.1f}%")
    print("=" * 85)
    print(f"  ★ GRAND TOTAL STEP TIME                     | {grand_total:>9.3f} ms | {'100.0%':>11} | {'100.0%':>11}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microsecond Fine-Grained CUDA LSTM Profiler")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence Length T (default: 64)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size N (default: 64)")
    parser.add_argument("--input_dim", type=int, default=128, help="Input Dim D (default: 128)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden Dim H (default: 256)")
    parser.add_argument("--runs", type=int, default=50, help="Profiling iterations (default: 50)")
    args = parser.parse_args()

    profile_lstm_breakdown(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_runs=args.runs,
    )

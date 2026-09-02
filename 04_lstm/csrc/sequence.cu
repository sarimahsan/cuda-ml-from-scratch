#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "kernels.cuh"

// In-place accumulation kernel: Accum [K x N] += Delta [K x N]
__global__ void accumulate_inplace_kernel(
    float* __restrict__ d_Accum,
    const float* __restrict__ d_Delta,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_Accum[idx] += d_Delta[idx];
    }
}

// -----------------------------------------------------------------------------
// HIGH-PERFORMANCE OPTIMIZED SEQUENCE FORWARD
// Fuses (G_ih + G_hh + b_hh) + 4-Gates + Cell State + Hidden State in-register
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> lstm_forward_sequence_fast(
    torch::Tensor X_seq,  // [T, N, D]
    torch::Tensor W_ih,   // [D, 4H]
    torch::Tensor b_ih,   // [4H]
    torch::Tensor W_hh,   // [H, 4H]
    torch::Tensor b_hh,   // [4H]
    torch::Tensor h_0,    // [N, H]
    torch::Tensor c_0     // [N, H]
) {
    TORCH_CHECK(X_seq.is_cuda() && W_ih.is_cuda() && W_hh.is_cuda(), "Inputs must be CUDA tensors");
    int T = X_seq.size(0);
    int N = X_seq.size(1);
    int D = X_seq.size(2);
    int four_H = W_ih.size(1);
    int H = four_H / 4;

    auto options = X_seq.options();

    // Allocate persistent sequence buffers
    auto H_seq = torch::empty({T, N, H}, options);
    auto C_seq = torch::empty({T + 1, N, H}, options);
    auto G_act_seq = torch::empty({T, N, four_H}, options);
    auto Tanh_C_seq = torch::empty({T, N, H}, options);

    // Initial state assignment
    C_seq[0].copy_(c_0);
    auto cur_h = h_0;

    // -------------------------------------------------------------------------
    // PILLAR 1: HIGH-THROUGHPUT PRECOMPUTED INPUT GEMM
    // [T*N, D] x [D, 4H] + b_ih -> [T*N, 4H] in a single hardware-tuned call
    // -------------------------------------------------------------------------
    auto X_flat = X_seq.reshape({T * N, D});
    torch::Tensor G_ih_all;
    if (b_ih.defined() && b_ih.numel() > 0) {
        G_ih_all = torch::addmm(b_ih, X_flat, W_ih);
    } else {
        G_ih_all = torch::mm(X_flat, W_ih);
    }

    // -------------------------------------------------------------------------
    // PILLAR 2: FUSED TEMPORAL LOOP
    // Evaluates Recurrent GEMM and Fused 4-Gates directly in registers
    // -------------------------------------------------------------------------
    int total_h_elements = N * H;
    int threads_1d = 256;
    int blocks_1d_gate = (total_h_elements + threads_1d - 1) / threads_1d;

    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;

    for (int t = 0; t < T; ++t) {
        float* d_g_ih_t = G_ih_all.data_ptr<float>() + t * N * four_H;
        float* d_c_prev = C_seq.data_ptr<float>() + t * N * H;
        float* d_c_next = C_seq.data_ptr<float>() + (t + 1) * N * H;
        float* d_h_next = H_seq.data_ptr<float>() + t * N * H;
        float* d_tanh_c = Tanh_C_seq.data_ptr<float>() + t * N * H;
        float* d_g_act = G_act_seq.data_ptr<float>() + t * N * four_H;

        // 1. Recurrent GEMM: G_hh = H_{t-1} [N x H] * W_hh [H x 4H]
        auto G_hh_t = torch::mm(cur_h, W_hh);

        // 2. Single Fused Step Kernel (G_ih + G_hh + b_hh -> 4-Gates -> Cell & Hidden state)
        fused_lstm_step_forward_kernel_torch<<<blocks_1d_gate, threads_1d>>>(
            d_g_ih_t,
            G_hh_t.data_ptr<float>(),
            b_hh_ptr,
            d_c_prev,
            d_g_act,
            d_c_next,
            d_tanh_c,
            d_h_next,
            N, H
        );

        cur_h = H_seq[t];
    }

    auto h_T = H_seq[T - 1];
    auto c_T = C_seq[T];

    return {H_seq, h_T, c_T, C_seq, G_act_seq, Tanh_C_seq, G_ih_all};
}

// -----------------------------------------------------------------------------
// HIGH-PERFORMANCE OPTIMIZED SEQUENCE BACKWARD (BPTT)
// In-place gradient accumulation & hardware-accelerated batched GEMMs
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> lstm_backward_sequence_fast(
    torch::Tensor dH_seq,    // [T, N, H]
    torch::Tensor X_seq,     // [T, N, D]
    torch::Tensor W_ih,      // [D, 4H]
    torch::Tensor W_hh,      // [H, 4H]
    torch::Tensor h_0,       // [N, H]
    torch::Tensor H_seq,     // [T, N, H]
    torch::Tensor C_seq,     // [T + 1, N, H]
    torch::Tensor G_act_seq, // [T, N, 4H]
    torch::Tensor Tanh_C_seq // [T, N, H]
) {
    TORCH_CHECK(dH_seq.is_cuda() && X_seq.is_cuda(), "Inputs must be CUDA tensors");
    int T = dH_seq.size(0);
    int N = dH_seq.size(1);
    int H = dH_seq.size(2);
    int D = X_seq.size(2);
    int four_H = 4 * H;
    auto options = dH_seq.options();

    auto dW_hh = torch::zeros({H, four_H}, options);
    auto db_hh = torch::zeros({four_H}, options);
    auto dG_all = torch::empty({T * N, four_H}, options);

    auto dh_next = torch::zeros({N, H}, options);
    auto dc_next = torch::zeros({N, H}, options);

    int total_h = N * H;
    int threads_1d = 256;
    int blocks_1d_gate = (total_h + threads_1d - 1) / threads_1d;

    // -------------------------------------------------------------------------
    // PILLAR 2: RECURRENT BPTT TEMPORAL LOOP
    // -------------------------------------------------------------------------
    for (int t = T - 1; t >= 0; --t) {
        float* d_dh_t = dH_seq.data_ptr<float>() + t * N * H;
        float* d_g_act_t = G_act_seq.data_ptr<float>() + t * N * four_H;
        float* d_c_prev = C_seq.data_ptr<float>() + t * N * H;
        float* d_c_curr = C_seq.data_ptr<float>() + (t + 1) * N * H;
        float* d_tanh_c = Tanh_C_seq.data_ptr<float>() + t * N * H;
        float* d_dg_t = dG_all.data_ptr<float>() + t * N * four_H;

        auto h_prev = (t > 0) ? H_seq[t - 1] : h_0;

        // 1. Accumulate dh_curr = dH_seq[t] + dh_next
        accumulate_inplace_kernel<<<blocks_1d_gate, threads_1d>>>(
            dh_next.data_ptr<float>(), d_dh_t, total_h
        );

        // 2. Fused Gate Backward Kernel
        fused_lstm_gates_backward_kernel_torch<<<blocks_1d_gate, threads_1d>>>(
            dh_next.data_ptr<float>(),
            dc_next.data_ptr<float>(),
            d_g_act_t,
            d_c_prev,
            d_c_curr,
            d_tanh_c,
            d_dg_t,
            dc_next.data_ptr<float>(),
            N, H
        );

        auto dg_t_tensor = dG_all.narrow(0, t * N, N);

        // 3. In-place dW_hh accumulation: dW_hh += h_prev^T * dG_t
        dW_hh.addmm_(h_prev.t(), dg_t_tensor);

        // 4. In-place db_hh accumulation: db_hh += sum(dG_t)
        db_hh.add_(dg_t_tensor.sum(0));

        // 5. Recurrent state gradient: dh_next = dG_t * W_hh^T
        dh_next = torch::mm(dg_t_tensor, W_hh.t());
    }

    // -------------------------------------------------------------------------
    // PILLAR 1: HIGH-THROUGHPUT BATCHED INPUT BACKWARD GEMMs
    // -------------------------------------------------------------------------
    auto X_flat = X_seq.reshape({T * N, D});

    // 1. dW_ih = X_all^T [D x T*N] * dG_all [T*N x 4H]
    auto dW_ih = torch::mm(X_flat.t(), dG_all);

    // 2. db_ih = sum(dG_all)
    auto db_ih = dG_all.sum(0);

    // 3. dX_all = dG_all [T*N x 4H] * W_ih^T [4H x D]
    auto dX_seq = torch::mm(dG_all, W_ih.t()).view({T, N, D});

    return {dW_ih, db_ih, dW_hh, db_hh, dX_seq};
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "kernels.cuh"

// Fast fused addition: G_total = G_ih + G_hh + b_hh
__global__ void add_fused_gates_kernel(
    const float* __restrict__ d_G_ih,
    const float* __restrict__ d_G_hh,
    const float* __restrict__ d_b_hh,
    float* __restrict__ d_G_tot,
    int total_elements,
    int four_H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % four_H;
        float val = d_G_ih[idx] + d_G_hh[idx];
        if (d_b_hh) val += d_b_hh[col];
        d_G_tot[idx] = val;
    }
}

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
// HIGH-PERFORMANCE NATIVE C++ SEQUENCE FORWARD
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
    auto G_ih_all = torch::empty({T * N, four_H}, options);
    auto G_hh_t = torch::empty({N, four_H}, options);
    auto G_tot_t = torch::empty({N, four_H}, options);

    // Copy initial states into C_seq[0] and H_list initial
    C_seq[0].copy_(c_0);
    auto cur_h = h_0;
    auto cur_c = c_0;

    // -------------------------------------------------------------------------
    // PILLAR 1: BATCHED INPUT GEMM (Pre-compute entire sequence in 1 large GEMM)
    // [T*N, D] x [D, 4H] -> [T*N, 4H]
    // -------------------------------------------------------------------------
    int total_tokens = T * N;
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_ih((four_H + TILE_DIM - 1) / TILE_DIM, (total_tokens + TILE_DIM - 1) / TILE_DIM);

    const float* b_ih_ptr = (b_ih.defined() && b_ih.numel() > 0) ? b_ih.data_ptr<float>() : nullptr;
    gemm_forward_kernel_torch<<<grid_ih, block_dim>>>(
        X_seq.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        b_ih_ptr,
        G_ih_all.data_ptr<float>(),
        total_tokens, D, four_H
    );

    // -------------------------------------------------------------------------
    // PILLAR 2: NATIVE C++ RECURRENT TEMPORAL LOOP
    // -------------------------------------------------------------------------
    dim3 grid_hh((four_H + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    int total_gate_elements = N * four_H;
    int threads_1d = 256;
    int blocks_1d_add = (total_gate_elements + threads_1d - 1) / threads_1d;
    int total_h_elements = N * H;
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
        gemm_forward_kernel_torch<<<grid_hh, block_dim>>>(
            cur_h.data_ptr<float>(),
            W_hh.data_ptr<float>(),
            nullptr,
            G_hh_t.data_ptr<float>(),
            N, H, four_H
        );

        // 2. Add G_total = G_ih + G_hh + b_hh
        add_fused_gates_kernel<<<blocks_1d_add, threads_1d>>>(
            d_g_ih_t,
            G_hh_t.data_ptr<float>(),
            b_hh_ptr,
            G_tot_t.data_ptr<float>(),
            total_gate_elements,
            four_H
        );

        // 3. Fused 4-Gate Activation Kernel
        fused_lstm_gates_forward_kernel_torch<<<blocks_1d_gate, threads_1d>>>(
            G_tot_t.data_ptr<float>(),
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
// HIGH-PERFORMANCE NATIVE C++ SEQUENCE BACKWARD (BPTT)
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

    auto dW_ih = torch::empty({D, four_H}, options);
    auto db_ih = torch::empty({four_H}, options);
    auto dW_hh = torch::zeros({H, four_H}, options);
    auto db_hh = torch::zeros({four_H}, options);
    auto dX_seq = torch::empty({T * N, D}, options);

    auto dG_all = torch::empty({T * N, four_H}, options);

    auto dh_curr = torch::empty({N, H}, options);
    auto dh_next = torch::zeros({N, H}, options);
    auto dc_next = torch::zeros({N, H}, options);

    auto dW_hh_step = torch::empty({H, four_H}, options);
    auto db_hh_step = torch::empty({four_H}, options);

    dim3 block_dim(TILE_DIM, TILE_DIM);
    int total_h = N * H;
    int threads_1d = 256;
    int blocks_1d_gate = (total_h + threads_1d - 1) / threads_1d;

    dim3 grid_w_hh((four_H + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);
    dim3 grid_h_prev((H + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    int blocks_b_hh = (four_H + threads_1d - 1) / threads_1d;
    int size_w_hh = H * four_H;

    // -------------------------------------------------------------------------
    // PILLAR 2: RECURRENT BPTT C++ TEMPORAL LOOP
    // -------------------------------------------------------------------------
    for (int t = T - 1; t >= 0; --t) {
        float* d_dh_t = dH_seq.data_ptr<float>() + t * N * H;
        float* d_g_act_t = G_act_seq.data_ptr<float>() + t * N * four_H;
        float* d_c_prev = C_seq.data_ptr<float>() + t * N * H;
        float* d_c_curr = C_seq.data_ptr<float>() + (t + 1) * N * H;
        float* d_tanh_c = Tanh_C_seq.data_ptr<float>() + t * N * H;
        float* d_dg_t = dG_all.data_ptr<float>() + t * N * four_H;

        float* d_h_prev = (t > 0) ? (H_seq.data_ptr<float>() + (t - 1) * N * H) : h_0.data_ptr<float>();

        // 1. Compute dh_curr = dH_seq[t] + dh_next
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

        // 3. dW_hh GEMM: H_prev^T [H x N] * dG_t [N x 4H]
        gemm_backward_weights_kernel_torch<<<grid_w_hh, block_dim>>>(
            d_h_prev,
            d_dg_t,
            dW_hh_step.data_ptr<float>(),
            N, H, four_H
        );
        accumulate_inplace_kernel<<<(size_w_hh + threads_1d - 1) / threads_1d, threads_1d>>>(
            dW_hh.data_ptr<float>(), dW_hh_step.data_ptr<float>(), size_w_hh
        );

        // 4. db_hh reduction: sum(dG_t)
        gemm_backward_bias_kernel_torch<<<blocks_b_hh, threads_1d>>>(
            d_dg_t,
            db_hh_step.data_ptr<float>(),
            N, four_H
        );
        accumulate_inplace_kernel<<<blocks_b_hh, threads_1d>>>(
            db_hh.data_ptr<float>(), db_hh_step.data_ptr<float>(), four_H
        );

        // 5. dH_prev GEMM: dG_t [N x 4H] * W_hh^T [4H x H] -> dh_next
        gemm_backward_data_kernel_torch<<<grid_h_prev, block_dim>>>(
            d_dg_t,
            W_hh.data_ptr<float>(),
            dh_next.data_ptr<float>(),
            N, H, four_H
        );
    }

    // -------------------------------------------------------------------------
    // PILLAR 1: BATCHED INPUT BACKWARD GEMMs
    // -------------------------------------------------------------------------
    int total_tokens = T * N;

    // 1. dW_ih = X_all^T [D x T*N] * dG_all [T*N x 4H] (1 large GEMM)
    dim3 grid_w_ih((four_H + TILE_DIM - 1) / TILE_DIM, (D + TILE_DIM - 1) / TILE_DIM);
    gemm_backward_weights_kernel_torch<<<grid_w_ih, block_dim>>>(
        X_seq.data_ptr<float>(),
        dG_all.data_ptr<float>(),
        dW_ih.data_ptr<float>(),
        total_tokens, D, four_H
    );

    // 2. db_ih = sum(dG_all) (1 single reduction)
    int blocks_b_ih = (four_H + threads_1d - 1) / threads_1d;
    gemm_backward_bias_kernel_torch<<<blocks_b_ih, threads_1d>>>(
        dG_all.data_ptr<float>(),
        db_ih.data_ptr<float>(),
        total_tokens, four_H
    );

    // 3. dX_all = dG_all [T*N x 4H] * W_ih^T [4H x D] (1 large GEMM)
    dim3 grid_x_all((D + TILE_DIM - 1) / TILE_DIM, (total_tokens + TILE_DIM - 1) / TILE_DIM);
    gemm_backward_data_kernel_torch<<<grid_x_all, block_dim>>>(
        dG_all.data_ptr<float>(),
        W_ih.data_ptr<float>(),
        dX_seq.data_ptr<float>(),
        total_tokens, D, four_H
    );

    auto dX_out = dX_seq.view({T, N, D});
    return {dW_ih, db_ih, dW_hh, db_hh, dX_out};
}

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace cuda_ml {
namespace gru {

// Forward declarations from gru_cell.cu
__global__ void accumulate_inplace_kernel(float* __restrict__ accum,
                                         const float* __restrict__ delta,
                                         int size);

void launch_gru_step_forward(
    const float* g_ih,
    const float* g_hh,
    const float* b_hh,
    const float* h_prev,
    float* gates_act,
    float* h_out,
    int N,
    int H,
    cudaStream_t stream
);

void launch_gru_step_backward(
    const float* dh_total,
    const float* h_prev,
    const float* gates_act,
    const float* g_hh,
    const float* b_hh,
    float* dg_ih,
    float* dg_hh,
    float* dh_prev_direct,
    int N,
    int H,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// 1. High-Performance GRU Sequence Forward Pass
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> gru_forward_sequence(
    torch::Tensor X_seq,  // [T, N, D]
    torch::Tensor W_ih,   // [D, 3H]
    torch::Tensor b_ih,   // [3H]
    torch::Tensor W_hh,   // [H, 3H]
    torch::Tensor b_hh,   // [3H]
    torch::Tensor h_0     // [N, H]
) {
    TORCH_CHECK(X_seq.is_cuda() && W_ih.is_cuda() && W_hh.is_cuda(), "Inputs must be CUDA tensors");
    int T = X_seq.size(0);
    int N = X_seq.size(1);
    int D = X_seq.size(2);
    int three_H = W_ih.size(1);
    int H = three_H / 3;

    auto options = X_seq.options();
    auto H_seq = torch::empty({T, N, H}, options);
    auto gates_act_seq = torch::empty({T, N, three_H}, options);
    auto G_hh_seq = torch::empty({T, N, three_H}, options);

    // 1. Precomputed Input GEMM: [T*N, D] x [D, 3H] + b_ih -> [T*N, 3H]
    auto X_flat = X_seq.reshape({T * N, D});
    torch::Tensor G_ih_all;
    if (b_ih.defined() && b_ih.numel() > 0) {
        G_ih_all = torch::addmm(b_ih, X_flat, W_ih);
    } else {
        G_ih_all = torch::mm(X_flat, W_ih);
    }

    // 2. Fused Recurrent Temporal Loop
    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;
    auto cur_h = h_0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int t = 0; t < T; ++t) {
        float* d_g_ih_t = G_ih_all.data_ptr<float>() + t * N * three_H;
        float* d_g_hh_t = G_hh_seq.data_ptr<float>() + t * N * three_H;
        float* d_gates_act_t = gates_act_seq.data_ptr<float>() + t * N * three_H;
        float* d_h_next = H_seq.data_ptr<float>() + t * N * H;

        // Recurrent projection: G_hh = H_{t-1} [N x H] * W_hh [H x 3H]
        auto G_hh_t = torch::mm(cur_h, W_hh);
        G_hh_seq[t].copy_(G_hh_t);

        // Fused GRU Step: computes (r, z, n) and blends h_t
        launch_gru_step_forward(
            d_g_ih_t,
            d_g_hh_t,
            b_hh_ptr,
            cur_h.data_ptr<float>(),
            d_gates_act_t,
            d_h_next,
            N,
            H,
            stream
        );

        cur_h = H_seq[t];
    }

    auto h_T = H_seq[T - 1];
    return {H_seq, h_T, gates_act_seq, G_hh_seq, G_ih_all};
}

// -----------------------------------------------------------------------------
// 2. High-Performance GRU Sequence Backward Pass (BPTT)
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> gru_backward_sequence(
    torch::Tensor dH_seq,       // [T, N, H]
    torch::Tensor X_seq,        // [T, N, D]
    torch::Tensor W_ih,         // [D, 3H]
    torch::Tensor b_hh,         // [3H]
    torch::Tensor W_hh,         // [H, 3H]
    torch::Tensor h_0,          // [N, H]
    torch::Tensor H_seq,        // [T, N, H]
    torch::Tensor gates_act_seq,// [T, N, 3H]
    torch::Tensor G_hh_seq      // [T, N, 3H]
) {
    TORCH_CHECK(dH_seq.is_cuda() && X_seq.is_cuda(), "Inputs must be CUDA tensors");
    int T = dH_seq.size(0);
    int N = dH_seq.size(1);
    int H = dH_seq.size(2);
    int D = X_seq.size(2);
    int three_H = 3 * H;

    auto options = dH_seq.options();
    auto dG_ih_all = torch::empty({T * N, three_H}, options);
    auto dG_hh_all = torch::empty({T * N, three_H}, options);
    auto dh_next = torch::zeros({N, H}, options);
    auto dh_prev_direct = torch::empty({N, H}, options);

    int total_h = N * H;
    int threads = 256;
    int blocks = (total_h + threads - 1) / threads;

    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int t = T - 1; t >= 0; --t) {
        float* d_dh_t = dH_seq.data_ptr<float>() + t * N * H;
        float* d_gates_act_t = gates_act_seq.data_ptr<float>() + t * N * three_H;
        float* d_g_hh_t = G_hh_seq.data_ptr<float>() + t * N * three_H;
        float* d_dg_ih_t = dG_ih_all.data_ptr<float>() + t * N * three_H;
        float* d_dg_hh_t = dG_hh_all.data_ptr<float>() + t * N * three_H;

        torch::Tensor h_prev_t = (t > 0) ? H_seq[t - 1] : h_0;

        // 1. Accumulate incoming gradient: dh_curr = dH_seq[t] + dh_next
        accumulate_inplace_kernel<<<blocks, threads, 0, stream>>>(
            dh_next.data_ptr<float>(), d_dh_t, total_h
        );

        // 2. Fused GRU step backward
        launch_gru_step_backward(
            dh_next.data_ptr<float>(),
            h_prev_t.data_ptr<float>(),
            d_gates_act_t,
            d_g_hh_t,
            b_hh_ptr,
            d_dg_ih_t,
            d_dg_hh_t,
            dh_prev_direct.data_ptr<float>(),
            N,
            H,
            stream
        );

        // 3. Recurrent state gradient: dh_next = dh_prev_direct + dg_hh_t * W_hh^T
        auto dg_hh_t_tensor = dG_hh_all.narrow(0, t * N, N);
        dh_next = dh_prev_direct + torch::mm(dg_hh_t_tensor, W_hh.t());
    }

    // Batched parameter reductions across all timesteps
    auto X_flat = X_seq.reshape({T * N, D});

    // dW_ih = X_all^T [D x T*N] * dG_ih_all [T*N x 3H]
    auto dW_ih = torch::mm(X_flat.t(), dG_ih_all);
    auto db_ih = dG_ih_all.sum(0);

    // dW_hh = H_prev_all^T [H x T*N] * dG_hh_all [T*N x 3H]
    auto H_prev_all = torch::cat({h_0.unsqueeze(0), H_seq.slice(0, 0, T - 1)}, 0).reshape({T * N, H});
    auto dW_hh = torch::mm(H_prev_all.t(), dG_hh_all);
    auto db_hh = dG_hh_all.sum(0);

    // dX_seq = dG_ih_all [T*N x 3H] * W_ih^T [3H x D]
    auto dX_seq = torch::mm(dG_ih_all, W_ih.t()).view({T, N, D});

    return {dW_ih, db_ih, dW_hh, db_hh, dX_seq};
}

} // namespace gru
} // namespace cuda_ml

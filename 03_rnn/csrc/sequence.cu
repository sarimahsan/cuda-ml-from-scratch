#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace cuda_ml {
namespace rnn {

// Forward declarations of device kernels in rnn_cell.cu
__global__ void accumulate_inplace_kernel(float* __restrict__ accum,
                                         const float* __restrict__ delta,
                                         int size);

__global__ void rnn_step_forward_kernel(const float* __restrict__ g_ih,
                                        const float* __restrict__ g_hh,
                                        const float* __restrict__ b_hh,
                                        float* __restrict__ h_out,
                                        int size,
                                        int H,
                                        int activation_type);

__global__ void rnn_step_backward_kernel(const float* __restrict__ dh_total,
                                         const float* __restrict__ h_t,
                                         float* __restrict__ dz_out,
                                         int size,
                                         int activation_type);

// -----------------------------------------------------------------------------
// 1. High-Performance Sequence Forward Pass
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> rnn_forward_sequence(
    torch::Tensor X_seq,  // [T, N, D]
    torch::Tensor W_ih,   // [D, H]
    torch::Tensor b_ih,   // [H]
    torch::Tensor W_hh,   // [H, H]
    torch::Tensor b_hh,   // [H]
    torch::Tensor h_0,    // [N, H]
    int activation_type   // 0 = tanh, 1 = relu
) {
    TORCH_CHECK(X_seq.is_cuda() && W_ih.is_cuda() && W_hh.is_cuda(), "Inputs must be CUDA tensors");
    int T = X_seq.size(0);
    int N = X_seq.size(1);
    int D = X_seq.size(2);
    int H = W_ih.size(1);

    auto options = X_seq.options();
    auto H_seq = torch::empty({T, N, H}, options);

    // 1. Precomputed Input Projections: [T*N, D] x [D, H] + b_ih -> [T*N, H]
    auto X_flat = X_seq.reshape({T * N, D});
    torch::Tensor G_ih_all;
    if (b_ih.defined() && b_ih.numel() > 0) {
        G_ih_all = torch::addmm(b_ih, X_flat, W_ih);
    } else {
        G_ih_all = torch::mm(X_flat, W_ih);
    }

    // 2. Fused Recurrent Temporal Loop
    int total_elements_per_step = N * H;
    int threads = 256;
    int blocks = (total_elements_per_step + threads - 1) / threads;

    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;
    auto cur_h = h_0;

    for (int t = 0; t < T; ++t) {
        float* d_g_ih_t = G_ih_all.data_ptr<float>() + t * N * H;
        float* d_h_next = H_seq.data_ptr<float>() + t * N * H;

        // Recurrent projection: G_hh = H_{t-1} [N x H] * W_hh [H x H]
        auto G_hh_t = torch::mm(cur_h, W_hh);

        // Fused elementwise step: h_t = act(g_ih_t + g_hh_t + b_hh)
        rnn_step_forward_kernel<<<blocks, threads>>>(
            d_g_ih_t,
            G_hh_t.data_ptr<float>(),
            b_hh_ptr,
            d_h_next,
            total_elements_per_step,
            H,
            activation_type
        );

        cur_h = H_seq[t];
    }

    auto h_T = H_seq[T - 1];
    return {H_seq, h_T, G_ih_all};
}

// -----------------------------------------------------------------------------
// 2. High-Performance Sequence Backward Pass (BPTT)
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> rnn_backward_sequence(
    torch::Tensor dH_seq,    // [T, N, H]
    torch::Tensor X_seq,     // [T, N, D]
    torch::Tensor W_ih,      // [D, H]
    torch::Tensor W_hh,      // [H, H]
    torch::Tensor h_0,       // [N, H]
    torch::Tensor H_seq,     // [T, N, H]
    int activation_type      // 0 = tanh, 1 = relu
) {
    TORCH_CHECK(dH_seq.is_cuda() && X_seq.is_cuda(), "Inputs must be CUDA tensors");
    int T = dH_seq.size(0);
    int N = dH_seq.size(1);
    int H = dH_seq.size(2);
    int D = X_seq.size(2);

    auto options = dH_seq.options();
    auto dZ_all = torch::empty({T * N, H}, options);
    auto dh_next = torch::zeros({N, H}, options);

    int total_elements_per_step = N * H;
    int threads = 256;
    int blocks = (total_elements_per_step + threads - 1) / threads;

    // Temporal BPTT Loop
    for (int t = T - 1; t >= 0; --t) {
        float* d_dh_t = dH_seq.data_ptr<float>() + t * N * H;
        float* d_h_t  = H_seq.data_ptr<float>() + t * N * H;
        float* d_dz_t = dZ_all.data_ptr<float>() + t * N * H;

        // 1. Accumulate incoming gradient: dh_curr = dH_seq[t] + dh_next
        accumulate_inplace_kernel<<<blocks, threads>>>(
            dh_next.data_ptr<float>(), d_dh_t, total_elements_per_step
        );

        // 2. Step backward: dz_t = dh_curr * d_act(h_t)
        rnn_step_backward_kernel<<<blocks, threads>>>(
            dh_next.data_ptr<float>(),
            d_h_t,
            d_dz_t,
            total_elements_per_step,
            activation_type
        );

        // 3. Recurrent state gradient: dh_next = dz_t * W_hh^T
        auto dz_t_tensor = dZ_all.narrow(0, t * N, N);
        dh_next = torch::mm(dz_t_tensor, W_hh.t());
    }

    // Batched Parameter Gradients across all timesteps
    auto X_flat = X_seq.reshape({T * N, D});

    // dW_ih = X_all^T [D x T*N] * dZ_all [T*N x H]
    auto dW_ih = torch::mm(X_flat.t(), dZ_all);

    // db_ih = sum(dZ_all, dim=0)
    auto db_ih = dZ_all.sum(0);
    auto db_hh = db_ih.clone();

    // dW_hh = H_prev_all^T [H x T*N] * dZ_all [T*N x H] in a single batched GEMM
    auto H_prev_all = torch::cat({h_0.unsqueeze(0), H_seq.slice(0, 0, T - 1)}, 0).reshape({T * N, H});
    auto dW_hh = torch::mm(H_prev_all.t(), dZ_all);

    // dX_seq = dZ_all [T*N x H] * W_ih^T [H x D]
    auto dX_seq = torch::mm(dZ_all, W_ih.t()).view({T, N, D});

    return {dW_ih, db_ih, dW_hh, db_hh, dX_seq};
}

} // namespace rnn
} // namespace cuda_ml

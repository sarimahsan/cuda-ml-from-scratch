#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../../kernels/include/gemm.cuh"
#include "../../kernels/include/softmax.cuh"
#include "../../kernels/include/optimizers.cuh"
#include "../../kernels/include/reduction.cuh"
#include "../../kernels/include/elementwise.cuh"

namespace cuda_ml {
namespace gru {

// Forward declarations from sequence.cu
std::vector<torch::Tensor> gru_forward_sequence(
    torch::Tensor X_seq, torch::Tensor W_ih, torch::Tensor b_ih,
    torch::Tensor W_hh, torch::Tensor b_hh, torch::Tensor h_0);

std::vector<torch::Tensor> gru_backward_sequence(
    torch::Tensor dH_seq, torch::Tensor X_seq, torch::Tensor W_ih,
    torch::Tensor b_hh, torch::Tensor W_hh, torch::Tensor h_0,
    torch::Tensor H_seq, torch::Tensor gates_act_seq, torch::Tensor G_hh_seq);

// Forward declarations from gru_cell.cu
void launch_gru_step_forward(
    const float* g_ih, const float* g_hh, const float* b_hh,
    const float* h_prev, float* gates_act, float* h_out,
    int N, int H, cudaStream_t stream);

void launch_gru_step_backward(
    const float* dh_total, const float* h_prev, const float* gates_act,
    const float* g_hh, const float* b_hh, float* dg_ih, float* dg_hh,
    float* dh_prev_direct, int N, int H, cudaStream_t stream);

} // namespace gru
} // namespace cuda_ml

// -----------------------------------------------------------------------------
// Single-Step GRU Cell Forward
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> gru_cell_forward_cuda(
    torch::Tensor x,      // [N, D]
    torch::Tensor h_prev, // [N, H]
    torch::Tensor W_ih,   // [D, 3H]
    torch::Tensor b_ih,   // [3H]
    torch::Tensor W_hh,   // [H, 3H]
    torch::Tensor b_hh    // [3H]
) {
    TORCH_CHECK(x.is_cuda() && h_prev.is_cuda() && W_ih.is_cuda() && W_hh.is_cuda(), "Inputs must be CUDA tensors");
    int N = x.size(0);
    int D = x.size(1);
    int three_H = W_ih.size(1);
    int H = three_H / 3;

    auto options = x.options();
    auto h_out = torch::empty({N, H}, options);
    auto gates_act = torch::empty({N, three_H}, options);
    auto g_ih = torch::empty({N, three_H}, options);
    auto g_hh = torch::empty({N, three_H}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. Projections
    cuda_ml::kernels::gemm_register_tiled(x.data_ptr<float>(), W_ih.data_ptr<float>(), g_ih.data_ptr<float>(), N, three_H, D, 1.0f, 0.0f, stream);
    if (b_ih.defined() && b_ih.numel() > 0) {
        cuda_ml::kernels::broadcast_bias_add(g_ih.data_ptr<float>(), b_ih.data_ptr<float>(), g_ih.data_ptr<float>(), N, three_H, stream);
    }
    cuda_ml::kernels::gemm_register_tiled(h_prev.data_ptr<float>(), W_hh.data_ptr<float>(), g_hh.data_ptr<float>(), N, three_H, H, 1.0f, 0.0f, stream);

    // 2. Fused GRU Step
    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;
    cuda_ml::gru::launch_gru_step_forward(
        g_ih.data_ptr<float>(),
        g_hh.data_ptr<float>(),
        b_hh_ptr,
        h_prev.data_ptr<float>(),
        gates_act.data_ptr<float>(),
        h_out.data_ptr<float>(),
        N,
        H,
        stream
    );

    return {h_out, gates_act, g_hh};
}

// -----------------------------------------------------------------------------
// Single-Step GRU Cell Backward
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> gru_cell_backward_cuda(
    torch::Tensor dh_t,       // [N, H]
    torch::Tensor x,          // [N, D]
    torch::Tensor h_prev,     // [N, H]
    torch::Tensor W_ih,       // [D, 3H]
    torch::Tensor W_hh,       // [H, 3H]
    torch::Tensor b_hh,       // [3H]
    torch::Tensor gates_act,  // [N, 3H]
    torch::Tensor g_hh        // [N, 3H]
) {
    int N = x.size(0);
    int D = x.size(1);
    int three_H = W_ih.size(1);
    int H = three_H / 3;

    auto options = x.options();
    auto dg_ih = torch::empty({N, three_H}, options);
    auto dg_hh = torch::empty({N, three_H}, options);
    auto dh_prev_direct = torch::empty({N, H}, options);
    auto dW_ih = torch::empty({D, three_H}, options);
    auto db_ih = torch::empty({three_H}, options);
    auto dW_hh = torch::empty({H, three_H}, options);
    auto db_hh = torch::empty({three_H}, options);
    auto dx = torch::empty({N, D}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;

    // 1. In-register step backward
    cuda_ml::gru::launch_gru_step_backward(
        dh_t.data_ptr<float>(),
        h_prev.data_ptr<float>(),
        gates_act.data_ptr<float>(),
        g_hh.data_ptr<float>(),
        b_hh_ptr,
        dg_ih.data_ptr<float>(),
        dg_hh.data_ptr<float>(),
        dh_prev_direct.data_ptr<float>(),
        N,
        H,
        stream
    );

    // 2. Gradients via centralized GEMMs
    // dW_ih = x^T * dg_ih
    cuda_ml::kernels::gemm_TN(x.data_ptr<float>(), dg_ih.data_ptr<float>(), dW_ih.data_ptr<float>(), D, three_H, N, 1.0f, 0.0f, stream);
    cuda_ml::kernels::reduce_col_sum(dg_ih.data_ptr<float>(), db_ih.data_ptr<float>(), N, three_H, stream);

    // dW_hh = h_prev^T * dg_hh
    cuda_ml::kernels::gemm_TN(h_prev.data_ptr<float>(), dg_hh.data_ptr<float>(), dW_hh.data_ptr<float>(), H, three_H, N, 1.0f, 0.0f, stream);
    cuda_ml::kernels::reduce_col_sum(dg_hh.data_ptr<float>(), db_hh.data_ptr<float>(), N, three_H, stream);

    // dx = dg_ih * W_ih^T
    cuda_ml::kernels::gemm_NT(dg_ih.data_ptr<float>(), W_ih.data_ptr<float>(), dx.data_ptr<float>(), N, D, three_H, 1.0f, 0.0f, stream);

    // dh_prev = dh_prev_direct + dg_hh * W_hh^T
    auto dh_prev = dh_prev_direct + torch::mm(dg_hh, W_hh.t());

    return {dW_ih, db_ih, dW_hh, db_hh, dx, dh_prev};
}

// -----------------------------------------------------------------------------
// Softmax Cross Entropy Wrapper
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> softmax_cross_entropy_cuda(torch::Tensor logits, torch::Tensor targets) {
    int N = logits.size(0);
    int C = logits.size(1);

    auto loss = torch::zeros({1}, logits.options());
    auto dLogits = torch::empty_like(logits);
    auto probs = torch::empty_like(logits);

    cuda_ml::kernels::fused_softmax_cross_entropy_forward_backward(
        logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(), dLogits.data_ptr<float>(), probs.data_ptr<float>(),
        N, C, at::cuda::getCurrentCUDAStream());

    return {probs, loss, dLogits};
}

// -----------------------------------------------------------------------------
// Optimizers
// -----------------------------------------------------------------------------
void adam_step_cuda(torch::Tensor param, torch::Tensor m, torch::Tensor v, torch::Tensor grad,
                    float lr, float beta1, float beta2, float eps, int step) {
    cuda_ml::kernels::adam_step(
        param.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), grad.data_ptr<float>(),
        lr, beta1, beta2, eps, step, param.numel(), 0.0f, at::cuda::getCurrentCUDAStream());
}

void sgd_momentum_step_cuda(torch::Tensor param, torch::Tensor velocity, torch::Tensor grad, float lr, float momentum) {
    cuda_ml::kernels::sgd_momentum_step(
        param.data_ptr<float>(), velocity.data_ptr<float>(), grad.data_ptr<float>(),
        lr, momentum, param.numel(), at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gru_cell_forward", &gru_cell_forward_cuda, "CUDA Single-Step GRU Cell Forward");
    m.def("gru_cell_backward", &gru_cell_backward_cuda, "CUDA Single-Step GRU Cell Backward Gradients");
    m.def("gru_forward_sequence", &cuda_ml::gru::gru_forward_sequence, "CUDA Fast Multi-Timestep GRU Sequence Forward");
    m.def("gru_backward_sequence", &cuda_ml::gru::gru_backward_sequence, "CUDA Fast Multi-Timestep GRU Sequence BPTT Backward");
    m.def("softmax_cross_entropy", &softmax_cross_entropy_cuda, "CUDA Fused Softmax + Cross Entropy Forward & Backward");
    m.def("adam_step", &adam_step_cuda, "CUDA In-Place Adam Optimizer Step");
    m.def("sgd_momentum_step", &sgd_momentum_step_cuda, "CUDA In-Place SGD with Momentum Optimizer Step");
}

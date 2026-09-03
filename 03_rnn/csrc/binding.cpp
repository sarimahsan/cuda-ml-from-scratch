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
namespace rnn {

// Forward declarations from sequence.cu
std::vector<torch::Tensor> rnn_forward_sequence(
    torch::Tensor X_seq, torch::Tensor W_ih, torch::Tensor b_ih,
    torch::Tensor W_hh, torch::Tensor b_hh, torch::Tensor h_0,
    int activation_type);

std::vector<torch::Tensor> rnn_backward_sequence(
    torch::Tensor dH_seq, torch::Tensor X_seq, torch::Tensor W_ih,
    torch::Tensor W_hh, torch::Tensor h_0, torch::Tensor H_seq,
    int activation_type);

// Forward declarations from rnn_cell.cu
void launch_rnn_step_forward(const float* g_ih, const float* g_hh, const float* b_hh,
                             float* h_out, int size, int H, int activation_type,
                             cudaStream_t stream);

void launch_rnn_step_backward(const float* dh_total, const float* h_t,
                              float* dz_out, int size, int activation_type,
                              cudaStream_t stream);

} // namespace rnn
} // namespace cuda_ml

// -----------------------------------------------------------------------------
// Single-Step RNN Cell Forward: h_t = act(x_t * W_ih + b_ih + h_{t-1} * W_hh + b_hh)
// -----------------------------------------------------------------------------
torch::Tensor rnn_cell_forward_cuda(
    torch::Tensor x,     // [N, D]
    torch::Tensor h_prev,// [N, H]
    torch::Tensor W_ih,  // [D, H]
    torch::Tensor b_ih,  // [H]
    torch::Tensor W_hh,  // [H, H]
    torch::Tensor b_hh,  // [H]
    int activation_type  // 0 = tanh, 1 = relu
) {
    TORCH_CHECK(x.is_cuda() && h_prev.is_cuda() && W_ih.is_cuda() && W_hh.is_cuda(), "Inputs must be CUDA tensors");
    int N = x.size(0);
    int D = x.size(1);
    int H = W_ih.size(1);

    auto options = x.options();
    auto h_out = torch::empty({N, H}, options);
    auto g_ih = torch::empty({N, H}, options);
    auto g_hh = torch::empty({N, H}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. Projections via high-throughput register-tiled GEMMs
    cuda_ml::kernels::gemm_register_tiled(x.data_ptr<float>(), W_ih.data_ptr<float>(), g_ih.data_ptr<float>(), N, H, D, 1.0f, 0.0f, stream);
    if (b_ih.defined() && b_ih.numel() > 0) {
        cuda_ml::kernels::broadcast_bias_add(g_ih.data_ptr<float>(), b_ih.data_ptr<float>(), g_ih.data_ptr<float>(), N, H, stream);
    }
    cuda_ml::kernels::gemm_register_tiled(h_prev.data_ptr<float>(), W_hh.data_ptr<float>(), g_hh.data_ptr<float>(), N, H, H, 1.0f, 0.0f, stream);

    // 2. Fused activation step
    int total_elements = N * H;
    const float* b_hh_ptr = (b_hh.defined() && b_hh.numel() > 0) ? b_hh.data_ptr<float>() : nullptr;

    cuda_ml::rnn::launch_rnn_step_forward(
        g_ih.data_ptr<float>(),
        g_hh.data_ptr<float>(),
        b_hh_ptr,
        h_out.data_ptr<float>(),
        total_elements,
        H,
        activation_type,
        stream
    );

    return h_out;
}

// -----------------------------------------------------------------------------
// Single-Step RNN Cell Backward
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> rnn_cell_backward_cuda(
    torch::Tensor dh_t,  // [N, H]
    torch::Tensor h_t,   // [N, H]
    torch::Tensor x,     // [N, D]
    torch::Tensor h_prev,// [N, H]
    torch::Tensor W_ih,  // [D, H]
    torch::Tensor W_hh,  // [H, H]
    int activation_type  // 0 = tanh, 1 = relu
) {
    int N = x.size(0);
    int D = x.size(1);
    int H = W_ih.size(1);

    auto options = x.options();
    auto dz_t = torch::empty({N, H}, options);
    auto dW_ih = torch::empty({D, H}, options);
    auto db_ih = torch::empty({H}, options);
    auto dW_hh = torch::empty({H, H}, options);
    auto db_hh = torch::empty({H}, options);
    auto dx = torch::empty({N, D}, options);
    auto dh_prev = torch::empty({N, H}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1. Elementwise step backward
    int total_elements = N * H;
    cuda_ml::rnn::launch_rnn_step_backward(
        dh_t.data_ptr<float>(),
        h_t.data_ptr<float>(),
        dz_t.data_ptr<float>(),
        total_elements,
        activation_type,
        stream
    );

    // 2. Gradients via centralized GEMMs
    // dW_ih = x^T * dz_t
    cuda_ml::kernels::gemm_TN(x.data_ptr<float>(), dz_t.data_ptr<float>(), dW_ih.data_ptr<float>(), D, H, N, 1.0f, 0.0f, stream);
    // db_ih = sum(dz_t, dim=0)
    cuda_ml::kernels::reduce_col_sum(dz_t.data_ptr<float>(), db_ih.data_ptr<float>(), N, H, stream);
    db_hh.copy_(db_ih);

    // dW_hh = h_prev^T * dz_t
    cuda_ml::kernels::gemm_TN(h_prev.data_ptr<float>(), dz_t.data_ptr<float>(), dW_hh.data_ptr<float>(), H, H, N, 1.0f, 0.0f, stream);

    // dx = dz_t * W_ih^T
    cuda_ml::kernels::gemm_NT(dz_t.data_ptr<float>(), W_ih.data_ptr<float>(), dx.data_ptr<float>(), N, D, H, 1.0f, 0.0f, stream);

    // dh_prev = dz_t * W_hh^T
    cuda_ml::kernels::gemm_NT(dz_t.data_ptr<float>(), W_hh.data_ptr<float>(), dh_prev.data_ptr<float>(), N, H, H, 1.0f, 0.0f, stream);

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
    m.def("rnn_cell_forward", &rnn_cell_forward_cuda, "CUDA Single-Step RNN Cell Forward");
    m.def("rnn_cell_backward", &rnn_cell_backward_cuda, "CUDA Single-Step RNN Cell Backward Gradients");
    m.def("rnn_forward_sequence", &cuda_ml::rnn::rnn_forward_sequence, "CUDA Fast Multi-Timestep Sequence Forward");
    m.def("rnn_backward_sequence", &cuda_ml::rnn::rnn_backward_sequence, "CUDA Fast Multi-Timestep Sequence BPTT Backward");
    m.def("softmax_cross_entropy", &softmax_cross_entropy_cuda, "CUDA Fused Softmax + Cross Entropy Forward & Backward");
    m.def("adam_step", &adam_step_cuda, "CUDA In-Place Adam Optimizer Step");
    m.def("sgd_momentum_step", &sgd_momentum_step_cuda, "CUDA In-Place SGD with Momentum Optimizer Step");
}

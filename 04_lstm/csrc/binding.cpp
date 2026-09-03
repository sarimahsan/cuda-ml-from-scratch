#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../../kernels/include/gemm.cuh"
#include "../../kernels/include/softmax.cuh"
#include "../../kernels/include/optimizers.cuh"
#include "../../kernels/include/reduction.cuh"
#include "../../kernels/include/elementwise.cuh"

// Forward declarations of individual gate modules
torch::Tensor input_gate_forward(torch::Tensor Z_i);
torch::Tensor input_gate_backward(torch::Tensor di, torch::Tensor i_val);

torch::Tensor forget_gate_forward(torch::Tensor Z_f);
torch::Tensor forget_gate_backward(torch::Tensor df, torch::Tensor f_val);

torch::Tensor candidate_gate_forward(torch::Tensor Z_g);
torch::Tensor candidate_gate_backward(torch::Tensor dg, torch::Tensor g_val);

torch::Tensor output_gate_forward(torch::Tensor Z_o);
torch::Tensor output_gate_backward(torch::Tensor do_t, torch::Tensor o_val);

// Cell State module
std::vector<torch::Tensor> cell_state_forward(
    torch::Tensor f_val,
    torch::Tensor c_prev,
    torch::Tensor i_val,
    torch::Tensor g_val,
    torch::Tensor o_val
);

std::vector<torch::Tensor> cell_state_backward(
    torch::Tensor dh,
    torch::Tensor dc_next,
    torch::Tensor o_val,
    torch::Tensor tanh_c,
    torch::Tensor c_prev,
    torch::Tensor f_val,
    torch::Tensor i_val,
    torch::Tensor g_val
);

// Fused Gates module
std::vector<torch::Tensor> fused_lstm_gates_forward(
    torch::Tensor gates_preact,
    torch::Tensor c_prev
);

std::vector<torch::Tensor> fused_lstm_gates_backward(
    torch::Tensor dh,
    torch::Tensor dc_next,
    torch::Tensor gates_act,
    torch::Tensor c_prev,
    torch::Tensor c_next,
    torch::Tensor tanh_c
);

// Native Fast C++ Sequence Module
std::vector<torch::Tensor> lstm_forward_sequence_fast(
    torch::Tensor X_seq,
    torch::Tensor W_ih,
    torch::Tensor b_ih,
    torch::Tensor W_hh,
    torch::Tensor b_hh,
    torch::Tensor h_0,
    torch::Tensor c_0
);

std::vector<torch::Tensor> lstm_backward_sequence_fast(
    torch::Tensor dH_seq,
    torch::Tensor X_seq,
    torch::Tensor W_ih,
    torch::Tensor W_hh,
    torch::Tensor h_0,
    torch::Tensor H_seq,
    torch::Tensor C_seq,
    torch::Tensor G_act_seq,
    torch::Tensor Tanh_C_seq
);

// -----------------------------------------------------------------------------
// Bridge to Centralized CUDA Kernel Engine
// -----------------------------------------------------------------------------

// Linear Forward: Z = A * B + bias (A: M x K, B: K x N)
torch::Tensor linear_forward(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_ml::kernels::gemm_tiled(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, 1.0f, 0.0f, stream);

    if (bias.defined() && bias.numel() > 0) {
        cuda_ml::kernels::broadcast_bias_add(C.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, N, stream);
    }
    return C;
}

// Linear Backward: dW = A^T * dY, db = sum(dY, dim=0), dX = dY * B^T
std::vector<torch::Tensor> linear_backward(torch::Tensor dY, torch::Tensor A, torch::Tensor B, bool compute_dX) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto dW = torch::empty({K, N}, A.options());
    auto db = torch::empty({N}, A.options());
    torch::Tensor dX;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_ml::kernels::gemm_TN(A.data_ptr<float>(), dY.data_ptr<float>(), dW.data_ptr<float>(), K, N, M, 1.0f, 0.0f, stream);
    cuda_ml::kernels::reduce_col_sum(dY.data_ptr<float>(), db.data_ptr<float>(), M, N, stream);

    if (compute_dX) {
        dX = torch::empty({M, K}, A.options());
        cuda_ml::kernels::gemm_NT(dY.data_ptr<float>(), B.data_ptr<float>(), dX.data_ptr<float>(), M, K, N, 1.0f, 0.0f, stream);
    }
    return {dW, db, dX};
}

torch::Tensor gemm_backward_weights(torch::Tensor A, torch::Tensor dY) {
    int M = A.size(0);
    int K = A.size(1);
    int N = dY.size(1);
    auto dW = torch::empty({K, N}, A.options());
    cuda_ml::kernels::gemm_TN(A.data_ptr<float>(), dY.data_ptr<float>(), dW.data_ptr<float>(), K, N, M, 1.0f, 0.0f, at::cuda::getCurrentCUDAStream());
    return dW;
}

torch::Tensor gemm_backward_bias(torch::Tensor dY) {
    int M = dY.size(0);
    int N = dY.size(1);
    auto db = torch::empty({N}, dY.options());
    cuda_ml::kernels::reduce_col_sum(dY.data_ptr<float>(), db.data_ptr<float>(), M, N, at::cuda::getCurrentCUDAStream());
    return db;
}

torch::Tensor gemm_backward_data(torch::Tensor dY, torch::Tensor W) {
    int M = dY.size(0);
    int N = dY.size(1);
    int K = W.size(0);
    auto dX = torch::empty({M, K}, dY.options());
    cuda_ml::kernels::gemm_NT(dY.data_ptr<float>(), W.data_ptr<float>(), dX.data_ptr<float>(), M, K, N, 1.0f, 0.0f, at::cuda::getCurrentCUDAStream());
    return dX;
}

// Softmax Cross Entropy
std::vector<torch::Tensor> softmax_cross_entropy(torch::Tensor logits, torch::Tensor targets) {
    int N = logits.size(0);
    int C = logits.size(1);

    auto loss = torch::zeros({1}, logits.options());
    auto dLogits = torch::empty_like(logits);

    cuda_ml::kernels::fused_softmax_cross_entropy_forward_backward(
        logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(), dLogits.data_ptr<float>(),
        N, C, at::cuda::getCurrentCUDAStream());

    return {loss, dLogits};
}

// Optimizers
void adam_step(torch::Tensor weight, torch::Tensor m, torch::Tensor v, torch::Tensor grad,
               float lr, float beta1, float beta2, float eps, int step) {
    cuda_ml::kernels::adam_step(
        weight.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), grad.data_ptr<float>(),
        lr, beta1, beta2, eps, step, weight.numel(), 0.0f, at::cuda::getCurrentCUDAStream());
}

void sgd_momentum_step(torch::Tensor weight, torch::Tensor v, torch::Tensor grad, float lr, float momentum) {
    cuda_ml::kernels::sgd_momentum_step(
        weight.data_ptr<float>(), v.data_ptr<float>(), grad.data_ptr<float>(),
        lr, momentum, weight.numel(), at::cuda::getCurrentCUDAStream());
}

void clip_grad_norm(std::vector<torch::Tensor> grads, float max_norm) {
    if (grads.empty()) return;

    float total_norm_sq = 0.0f;
    for (auto& g : grads) {
        if (g.defined() && g.numel() > 0) {
            total_norm_sq += g.norm(2).item<float>() * g.norm(2).item<float>();
        }
    }
    float total_norm = std::sqrt(total_norm_sq);

    if (total_norm > max_norm && total_norm > 0.0f) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto& g : grads) {
            if (g.defined() && g.numel() > 0) {
                g.mul_(scale);
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Modular Pure CUDA LSTM Extension powered by Centralized CUDA Kernel Engine";

    // 1. Dedicated Individual Gate functions
    m.def("input_gate_forward", &input_gate_forward, "Input Gate Forward (CUDA)");
    m.def("input_gate_backward", &input_gate_backward, "Input Gate Backward (CUDA)");

    m.def("forget_gate_forward", &forget_gate_forward, "Forget Gate Forward (CUDA)");
    m.def("forget_gate_backward", &forget_gate_backward, "Forget Gate Backward (CUDA)");

    m.def("candidate_gate_forward", &candidate_gate_forward, "Cell Candidate Gate Forward (CUDA)");
    m.def("candidate_gate_backward", &candidate_gate_backward, "Cell Candidate Gate Backward (CUDA)");

    m.def("output_gate_forward", &output_gate_forward, "Output Gate Forward (CUDA)");
    m.def("output_gate_backward", &output_gate_backward, "Output Gate Backward (CUDA)");

    // 2. Cell State & Hidden State updates
    m.def("cell_state_forward", &cell_state_forward, "Cell State Forward (CUDA)");
    m.def("cell_state_backward", &cell_state_backward, "Cell State Backward (CUDA)");

    // 3. Fused 4-Gate execution
    m.def("fused_lstm_gates_forward", &fused_lstm_gates_forward, "Fused LSTM Gates Forward (CUDA)");
    m.def("fused_lstm_gates_backward", &fused_lstm_gates_backward, "Fused LSTM Gates Backward (CUDA)");

    // 4. Linear & GEMMs (using shared kernel engine)
    m.def("linear_forward", &linear_forward, "2D Tiled Shared-Memory GEMM Forward (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("bias") = torch::Tensor());
    m.def("linear_backward", &linear_backward, "2D Tiled Shared-Memory GEMM Backward (CUDA)",
          py::arg("dY"), py::arg("A"), py::arg("B"), py::arg("compute_dX") = true);
    m.def("gemm_backward_weights", &gemm_backward_weights, "GEMM Backward dW (CUDA)");
    m.def("gemm_backward_bias", &gemm_backward_bias, "GEMM Backward db (CUDA)");
    m.def("gemm_backward_data", &gemm_backward_data, "GEMM Backward dX (CUDA)");

    // 5. Native Fast C++ Sequence Execution
    m.def("lstm_forward_sequence_fast", &lstm_forward_sequence_fast, "Fast Native C++ LSTM Sequence Forward (CUDA)");
    m.def("lstm_backward_sequence_fast", &lstm_backward_sequence_fast, "Fast Native C++ LSTM Sequence Backward (CUDA)");

    // 6. Loss & Optimizers
    m.def("softmax_cross_entropy", &softmax_cross_entropy, "Sequence Softmax & Cross-Entropy Loss with Warp Reductions (CUDA)");
    m.def("adam_step", &adam_step, "Vectorized In-Place Adam Step (CUDA)");
    m.def("sgd_momentum_step", &sgd_momentum_step, "Vectorized In-Place SGD Momentum Step (CUDA)");
    m.def("clip_grad_norm", &clip_grad_norm, "GPU In-Place Gradient Norm Clipping (CUDA)");
}

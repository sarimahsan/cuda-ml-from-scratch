#include <torch/extension.h>
#include <vector>

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

// Linear / GEMM module
torch::Tensor linear_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
);

std::vector<torch::Tensor> linear_backward(
    torch::Tensor dY,
    torch::Tensor A,
    torch::Tensor B,
    bool compute_dX
);

// Softmax Cross Entropy module
std::vector<torch::Tensor> softmax_cross_entropy(
    torch::Tensor logits,
    torch::Tensor targets
);

// Optimizer modules
void adam_step(
    torch::Tensor weight,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step
);

void sgd_momentum_step(
    torch::Tensor weight,
    torch::Tensor v,
    torch::Tensor grad,
    float lr,
    float momentum
);

void clip_grad_norm(
    std::vector<torch::Tensor> grads,
    float max_norm
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Modular Pure CUDA LSTM Extension with Separate Gate Kernels & Fused Operators";

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

    // 4. Linear & GEMMs
    m.def("linear_forward", &linear_forward, "2D Tiled Shared-Memory GEMM Forward (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("bias") = torch::Tensor());
    m.def("linear_backward", &linear_backward, "2D Tiled Shared-Memory GEMM Backward (CUDA)",
          py::arg("dY"), py::arg("A"), py::arg("B"), py::arg("compute_dX") = true);

    // 5. Loss
    m.def("softmax_cross_entropy", &softmax_cross_entropy, "Sequence Softmax & Cross-Entropy Loss with Warp Reductions (CUDA)");

    // 6. Optimizers & Clipping
    m.def("adam_step", &adam_step, "Vectorized In-Place Adam Step (CUDA)");
    m.def("sgd_momentum_step", &sgd_momentum_step, "Vectorized In-Place SGD Momentum Step (CUDA)");
    m.def("clip_grad_norm", &clip_grad_norm, "GPU In-Place Gradient Norm Clipping (CUDA)");
}

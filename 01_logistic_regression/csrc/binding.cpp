#include <torch/extension.h>

// Forward declarations of CUDA interface functions
torch::Tensor forward_cuda(torch::Tensor X, torch::Tensor w, torch::Tensor b);
torch::Tensor bce_loss_cuda(torch::Tensor y_hat, torch::Tensor y);
std::vector<torch::Tensor> backward_cuda(torch::Tensor X, torch::Tensor y_hat, torch::Tensor y);
void sgd_step_cuda(torch::Tensor w, torch::Tensor b, torch::Tensor grad_w, torch::Tensor grad_b, float lr);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Logistic Regression Forward Pass (CUDA)");
    m.def("bce_loss", &bce_loss_cuda, "Binary Cross Entropy Loss (CUDA)");
    m.def("backward", &backward_cuda, "Logistic Regression Backward Gradients (CUDA)");
    m.def("sgd_step", &sgd_step_cuda, "SGD Parameter Update (CUDA)");
}

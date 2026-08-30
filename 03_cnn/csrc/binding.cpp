#include <torch/extension.h>
#include <vector>

// Forward declarations of Conv2D functions (csrc/conv2d.cu)
torch::Tensor conv2d_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b, int stride, int pad);
std::vector<torch::Tensor> conv2d_backward_cuda(torch::Tensor dO, torch::Tensor X, torch::Tensor W, int stride, int pad, bool compute_dX);

// Forward declarations of MaxPool2D functions (csrc/pool.cu)
std::vector<torch::Tensor> maxpool2d_forward_cuda(torch::Tensor X, int pool_h, int pool_w, int stride, int pad);
torch::Tensor maxpool2d_backward_cuda(torch::Tensor dP, torch::Tensor mask, int N, int C, int H_in, int W_in);

// Forward declarations of Linear functions (csrc/linear.cu)
torch::Tensor linear_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b);
std::vector<torch::Tensor> linear_backward_cuda(torch::Tensor dZ, torch::Tensor X, torch::Tensor W, bool compute_dX);

// Forward declarations of Activation functions (csrc/activations.cu)
torch::Tensor relu_forward_cuda(torch::Tensor Z);
torch::Tensor relu_backward_cuda(torch::Tensor dA, torch::Tensor Z);
torch::Tensor gelu_forward_cuda(torch::Tensor Z);
torch::Tensor gelu_backward_cuda(torch::Tensor dA, torch::Tensor Z);
torch::Tensor sigmoid_forward_cuda(torch::Tensor Z);
torch::Tensor sigmoid_backward_cuda(torch::Tensor dA, torch::Tensor Z);

// Forward declarations of Softmax & Loss functions (csrc/softmax_loss.cu)
std::vector<torch::Tensor> softmax_cross_entropy_cuda(torch::Tensor logits, torch::Tensor targets);

// Forward declarations of Optimizer functions (csrc/optimizers.cu)
void sgd_momentum_step_cuda(torch::Tensor param, torch::Tensor velocity, torch::Tensor grad, float lr, float momentum);
void adam_step_cuda(torch::Tensor param, torch::Tensor m, torch::Tensor v, torch::Tensor grad, float lr, float beta1, float beta2, float eps, int step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Conv2D Layers
    m.def("conv2d_forward", &conv2d_forward_cuda, "CUDA Conv2D Forward");
    m.def("conv2d_backward", &conv2d_backward_cuda, "CUDA Conv2D Backward Gradients (dW, db, dX)");

    // MaxPool2D Layers
    m.def("maxpool2d_forward", &maxpool2d_forward_cuda, "CUDA MaxPool2D Forward (Output & Argmax Mask)");
    m.def("maxpool2d_backward", &maxpool2d_backward_cuda, "CUDA MaxPool2D Backward Routing");

    // Linear / GEMM Layers
    m.def("linear_forward", &linear_forward_cuda, "CUDA Tiled Linear Forward GEMM (Z = X * W + b)");
    m.def("linear_backward", &linear_backward_cuda, "CUDA Linear Backward Gradients (dW, db, dX)");

    // Activations
    m.def("relu_forward", &relu_forward_cuda, "CUDA ReLU Forward");
    m.def("relu_backward", &relu_backward_cuda, "CUDA ReLU Backward");
    m.def("gelu_forward", &gelu_forward_cuda, "CUDA GELU Forward");
    m.def("gelu_backward", &gelu_backward_cuda, "CUDA GELU Backward");
    m.def("sigmoid_forward", &sigmoid_forward_cuda, "CUDA Sigmoid Forward");
    m.def("sigmoid_backward", &sigmoid_backward_cuda, "CUDA Sigmoid Backward");

    // Softmax & Cross Entropy Loss
    m.def("softmax_cross_entropy", &softmax_cross_entropy_cuda, "CUDA Softmax + Categorical Cross Entropy Forward & Backward");

    // Optimizers
    m.def("sgd_momentum_step", &sgd_momentum_step_cuda, "CUDA In-Place SGD with Momentum Optimizer Step");
    m.def("adam_step", &adam_step_cuda, "CUDA In-Place Adam Optimizer Step");
}

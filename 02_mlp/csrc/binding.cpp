#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../../kernels/include/gemm.cuh"
#include "../../kernels/include/activation.cuh"
#include "../../kernels/include/softmax.cuh"
#include "../../kernels/include/optimizers.cuh"
#include "../../kernels/include/reduction.cuh"
#include "../../kernels/include/elementwise.cuh"

// Linear Forward: Z = X * W + b
torch::Tensor linear_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(1);

    auto Z = torch::empty({M, N}, X.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_ml::kernels::gemm_register_tiled(X.data_ptr<float>(), W.data_ptr<float>(), Z.data_ptr<float>(), M, N, K, 1.0f, 0.0f, stream);
    cuda_ml::kernels::broadcast_bias_add(Z.data_ptr<float>(), b.data_ptr<float>(), Z.data_ptr<float>(), M, N, stream);
    return Z;
}

// Linear Backward: dW = X^T * dZ, db = sum(dZ, dim=0), dX = dZ * W^T
std::vector<torch::Tensor> linear_backward_cuda(torch::Tensor dZ, torch::Tensor X, torch::Tensor W, bool compute_dX) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(1);

    auto dW = torch::empty({K, N}, X.options());
    auto db = torch::empty({N}, X.options());
    torch::Tensor dX;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dW = X^T * dZ
    cuda_ml::kernels::gemm_TN(X.data_ptr<float>(), dZ.data_ptr<float>(), dW.data_ptr<float>(), K, N, M, 1.0f, 0.0f, stream);

    // db = sum(dZ, dim=0)
    cuda_ml::kernels::reduce_col_sum(dZ.data_ptr<float>(), db.data_ptr<float>(), M, N, stream);

    if (compute_dX) {
        dX = torch::empty({M, K}, X.options());
        // dX = dZ * W^T
        cuda_ml::kernels::gemm_NT(dZ.data_ptr<float>(), W.data_ptr<float>(), dX.data_ptr<float>(), M, K, N, 1.0f, 0.0f, stream);
    }

    return {dW, db, dX};
}

// Activations
torch::Tensor relu_forward_cuda(torch::Tensor Z) {
    auto A = torch::empty_like(Z);
    cuda_ml::kernels::relu_forward(Z.data_ptr<float>(), A.data_ptr<float>(), Z.numel(), at::cuda::getCurrentCUDAStream());
    return A;
}

torch::Tensor relu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    auto dZ = torch::empty_like(dA);
    cuda_ml::kernels::relu_backward(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), dA.numel(), at::cuda::getCurrentCUDAStream());
    return dZ;
}

torch::Tensor gelu_forward_cuda(torch::Tensor Z) {
    auto A = torch::empty_like(Z);
    cuda_ml::kernels::gelu_forward(Z.data_ptr<float>(), A.data_ptr<float>(), Z.numel(), at::cuda::getCurrentCUDAStream());
    return A;
}

torch::Tensor gelu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    auto dZ = torch::empty_like(dA);
    cuda_ml::kernels::gelu_backward(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), dA.numel(), at::cuda::getCurrentCUDAStream());
    return dZ;
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor Z) {
    auto A = torch::empty_like(Z);
    cuda_ml::kernels::sigmoid_forward(Z.data_ptr<float>(), A.data_ptr<float>(), Z.numel(), at::cuda::getCurrentCUDAStream());
    return A;
}

torch::Tensor sigmoid_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    auto dZ = torch::empty_like(dA);
    cuda_ml::kernels::sigmoid_backward(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), dA.numel(), at::cuda::getCurrentCUDAStream());
    return dZ;
}

// Softmax Cross Entropy
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

// Optimizers
void sgd_momentum_step_cuda(torch::Tensor param, torch::Tensor velocity, torch::Tensor grad, float lr, float momentum) {
    cuda_ml::kernels::sgd_momentum_step(
        param.data_ptr<float>(), velocity.data_ptr<float>(), grad.data_ptr<float>(),
        lr, momentum, param.numel(), at::cuda::getCurrentCUDAStream());
}

void adam_step_cuda(torch::Tensor param, torch::Tensor m, torch::Tensor v, torch::Tensor grad,
                    float lr, float beta1, float beta2, float eps, int step) {
    cuda_ml::kernels::adam_step(
        param.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), grad.data_ptr<float>(),
        lr, beta1, beta2, eps, step, param.numel(), 0.0f, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward_cuda, "CUDA Register-Tiled Linear Forward GEMM (Z = X * W + b)");
    m.def("linear_backward", &linear_backward_cuda, "CUDA Linear Backward Gradients (dW, db, dX)");
    m.def("relu_forward", &relu_forward_cuda, "CUDA ReLU Forward");
    m.def("relu_backward", &relu_backward_cuda, "CUDA ReLU Backward");
    m.def("gelu_forward", &gelu_forward_cuda, "CUDA GELU Forward");
    m.def("gelu_backward", &gelu_backward_cuda, "CUDA GELU Backward");
    m.def("sigmoid_forward", &sigmoid_forward_cuda, "CUDA Sigmoid Forward");
    m.def("sigmoid_backward", &sigmoid_backward_cuda, "CUDA Sigmoid Backward");
    m.def("softmax_cross_entropy", &softmax_cross_entropy_cuda, "CUDA Softmax + Cross Entropy Forward & Backward");
    m.def("sgd_momentum_step", &sgd_momentum_step_cuda, "CUDA In-Place SGD with Momentum Optimizer Step");
    m.def("adam_step", &adam_step_cuda, "CUDA In-Place Adam Optimizer Step");
}

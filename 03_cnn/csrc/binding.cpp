#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../../kernels/include/convolution.cuh"
#include "../../kernels/include/pooling.cuh"
#include "../../kernels/include/gemm.cuh"
#include "../../kernels/include/activation.cuh"
#include "../../kernels/include/softmax.cuh"
#include "../../kernels/include/optimizers.cuh"
#include "../../kernels/include/reduction.cuh"

// Conv2D Forward
torch::Tensor conv2d_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b, int stride, int pad) {
    int N = X.size(0);
    int C_in = X.size(1);
    int H_in = X.size(2);
    int W_in = X.size(3);

    int C_out = W.size(0);
    int K_h = W.size(2);
    int K_w = W.size(3);

    int H_out = (H_in + 2 * pad - K_h) / stride + 1;
    int W_out = (W_in + 2 * pad - K_w) / stride + 1;

    auto Y = torch::empty({N, C_out, H_out, W_out}, X.options());
    cuda_ml::kernels::conv2d_direct_forward(
        X.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(), Y.data_ptr<float>(),
        N, C_in, H_in, W_in, C_out, K_h, K_w, pad, pad, stride, stride,
        at::cuda::getCurrentCUDAStream());
    return Y;
}

// Conv2D Backward
std::vector<torch::Tensor> conv2d_backward_cuda(torch::Tensor dO, torch::Tensor X, torch::Tensor W, int stride, int pad, bool compute_dX) {
    int N = X.size(0);
    int C_in = X.size(1);
    int H_in = X.size(2);
    int W_in = X.size(3);

    int C_out = W.size(0);
    int K_h = W.size(2);
    int K_w = W.size(3);

    int H_out = dO.size(2);
    int W_out = dO.size(3);

    auto dW = torch::zeros_like(W);
    auto db = torch::zeros({C_out}, X.options());
    torch::Tensor dX;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (compute_dX) {
        dX = torch::zeros_like(X);
    }

    return {dW, db, dX};
}

// MaxPool2D Forward
std::vector<torch::Tensor> maxpool2d_forward_cuda(torch::Tensor X, int pool_h, int pool_w, int stride, int pad) {
    int N = X.size(0);
    int C = X.size(1);
    int H_in = X.size(2);
    int W_in = X.size(3);

    int H_out = (H_in + 2 * pad - pool_h) / stride + 1;
    int W_out = (W_in + 2 * pad - pool_w) / stride + 1;

    auto Y = torch::empty({N, C, H_out, W_out}, X.options());
    auto mask = torch::empty({N, C, H_out, W_out}, X.options().dtype(torch::kInt64));

    cuda_ml::kernels::maxpool2d_forward(
        X.data_ptr<float>(), Y.data_ptr<float>(), mask.data_ptr<int64_t>(),
        N, C, H_in, W_in, pool_h, pool_w, stride, stride, pad, pad,
        at::cuda::getCurrentCUDAStream());

    return {Y, mask};
}

// MaxPool2D Backward
torch::Tensor maxpool2d_backward_cuda(torch::Tensor dP, torch::Tensor mask, int N, int C, int H_in, int W_in) {
    int H_out = dP.size(2);
    int W_out = dP.size(3);

    auto dX = torch::zeros({N, C, H_in, W_in}, dP.options());
    cuda_ml::kernels::maxpool2d_backward(
        dP.data_ptr<float>(), mask.data_ptr<int64_t>(), dX.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, at::cuda::getCurrentCUDAStream());
    return dX;
}

// Linear Forward: Z = X * W + b
torch::Tensor linear_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor b) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(1);

    auto Z = torch::empty({M, N}, X.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cuda_ml::kernels::gemm_tiled(X.data_ptr<float>(), W.data_ptr<float>(), Z.data_ptr<float>(), M, N, K, 1.0f, 0.0f, stream);
    cuda_ml::kernels::broadcast_bias_add(Z.data_ptr<float>(), b.data_ptr<float>(), Z.data_ptr<float>(), M, N, stream);
    return Z;
}

// Linear Backward
std::vector<torch::Tensor> linear_backward_cuda(torch::Tensor dZ, torch::Tensor X, torch::Tensor W, bool compute_dX) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(1);

    auto dW = torch::empty({K, N}, X.options());
    auto db = torch::empty({N}, X.options());
    torch::Tensor dX;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_ml::kernels::gemm_TN(X.data_ptr<float>(), dZ.data_ptr<float>(), dW.data_ptr<float>(), K, N, M, 1.0f, 0.0f, stream);
    cuda_ml::kernels::reduce_col_sum(dZ.data_ptr<float>(), db.data_ptr<float>(), M, N, stream);

    if (compute_dX) {
        dX = torch::empty({M, K}, X.options());
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

    cuda_ml::kernels::fused_softmax_cross_entropy_forward_backward(
        logits.data_ptr<float>(), targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(), dLogits.data_ptr<float>(),
        N, C, at::cuda::getCurrentCUDAStream());

    return {loss, dLogits};
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
    m.def("conv2d_forward", &conv2d_forward_cuda, "CUDA Conv2D Forward");
    m.def("conv2d_backward", &conv2d_backward_cuda, "CUDA Conv2D Backward Gradients");
    m.def("maxpool2d_forward", &maxpool2d_forward_cuda, "CUDA MaxPool2D Forward");
    m.def("maxpool2d_backward", &maxpool2d_backward_cuda, "CUDA MaxPool2D Backward");
    m.def("linear_forward", &linear_forward_cuda, "CUDA Tiled Linear Forward GEMM");
    m.def("linear_backward", &linear_backward_cuda, "CUDA Linear Backward Gradients");
    m.def("relu_forward", &relu_forward_cuda, "CUDA ReLU Forward");
    m.def("relu_backward", &relu_backward_cuda, "CUDA ReLU Backward");
    m.def("gelu_forward", &gelu_forward_cuda, "CUDA GELU Forward");
    m.def("gelu_backward", &gelu_backward_cuda, "CUDA GELU Backward");
    m.def("sigmoid_forward", &sigmoid_forward_cuda, "CUDA Sigmoid Forward");
    m.def("sigmoid_backward", &sigmoid_backward_cuda, "CUDA Sigmoid Backward");
    m.def("softmax_cross_entropy", &softmax_cross_entropy_cuda, "CUDA Softmax + Cross Entropy");
    m.def("sgd_momentum_step", &sgd_momentum_step_cuda, "CUDA SGD with Momentum");
    m.def("adam_step", &adam_step_cuda, "CUDA Adam Step");
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../include/gemm.cuh"
#include "../include/convolution.cuh"
#include "../include/reduction.cuh"
#include "../include/softmax.cuh"
#include "../include/normalization.cuh"
#include "../include/activation.cuh"
#include "../include/pooling.cuh"
#include "../include/elementwise.cuh"

// PyTorch Wrappers for GEMMs
torch::Tensor gemm_tiled_torch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner matrix dimensions must agree");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    cuda_ml::kernels::gemm_register_tiled(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, at::cuda::getCurrentCUDAStream());
    return C;
}

torch::Tensor gemm_register_tiled_torch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    cuda_ml::kernels::gemm_register_tiled(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f, at::cuda::getCurrentCUDAStream());
    return C;
}

// PyTorch Wrappers for Softmax
torch::Tensor softmax_forward_torch(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda(), "Logits must be CUDA tensor");
    auto probs = torch::empty_like(logits);
    int N = logits.size(0);
    int C = logits.size(1);
    cuda_ml::kernels::softmax_forward(
        logits.data_ptr<float>(), probs.data_ptr<float>(), N, C, at::cuda::getCurrentCUDAStream());
    return probs;
}

torch::Tensor online_safe_softmax_torch(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda(), "Logits must be CUDA tensor");
    auto probs = torch::empty_like(logits);
    int N = logits.size(0);
    int C = logits.size(1);
    cuda_ml::kernels::online_safe_softmax_forward(
        logits.data_ptr<float>(), probs.data_ptr<float>(), N, C, at::cuda::getCurrentCUDAStream());
    return probs;
}

// PyTorch Wrappers for LayerNorm
std::vector<torch::Tensor> layernorm_forward_torch(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, float eps) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA tensor");
    int N = X.size(0);
    int D = X.size(1);

    auto Y = torch::empty_like(X);
    auto mean = torch::empty({N}, X.options());
    auto rstd = torch::empty({N}, X.options());

    cuda_ml::kernels::layernorm_forward(
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        Y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N, D, eps, at::cuda::getCurrentCUDAStream());

    return {Y, mean, rstd};
}

// PyTorch Wrappers for Activations
torch::Tensor gelu_forward_torch(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA tensor");
    auto Y = torch::empty_like(X);
    cuda_ml::kernels::gelu_forward(X.data_ptr<float>(), Y.data_ptr<float>(), X.numel(), at::cuda::getCurrentCUDAStream());
    return Y;
}

torch::Tensor silu_forward_torch(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA tensor");
    auto Y = torch::empty_like(X);
    cuda_ml::kernels::silu_forward(X.data_ptr<float>(), Y.data_ptr<float>(), X.numel(), at::cuda::getCurrentCUDAStream());
    return Y;
}

// PyTorch Wrappers for Elementwise
torch::Tensor fused_residual_add_torch(torch::Tensor X, torch::Tensor residual) {
    TORCH_CHECK(X.is_cuda() && residual.is_cuda(), "Inputs must be CUDA tensors");
    auto Y = torch::empty_like(X);
    cuda_ml::kernels::fused_residual_add(
        X.data_ptr<float>(), residual.data_ptr<float>(), Y.data_ptr<float>(),
        X.numel(), at::cuda::getCurrentCUDAStream());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_tiled", &gemm_tiled_torch, "2D Shared Memory Tiled GEMM (CUDA)");
    m.def("gemm_register_tiled", &gemm_register_tiled_torch, "Register Tiled GEMM (CUDA)");
    m.def("softmax_forward", &softmax_forward_torch, "Warp-Level Softmax (CUDA)");
    m.def("online_safe_softmax", &online_safe_softmax_torch, "Online Safe FlashSoftmax (CUDA)");
    m.def("layernorm_forward", &layernorm_forward_torch, "LayerNorm Forward (CUDA)");
    m.def("gelu_forward", &gelu_forward_torch, "Vectorized GELU Forward (CUDA)");
    m.def("silu_forward", &silu_forward_torch, "Vectorized SiLU Forward (CUDA)");
    m.def("fused_residual_add", &fused_residual_add_torch, "Fused Residual Add (CUDA)");
}

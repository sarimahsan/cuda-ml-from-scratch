#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608f
#define COEFF 0.044715f

// -------------------------------------------------------------------------
// Forward Activation Kernels
// -------------------------------------------------------------------------
__global__ void relu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = Z[idx];
        A[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void gelu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        A[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 1.0f / (1.0f + __expf(-Z[idx]));
    }
}

// -------------------------------------------------------------------------
// Backward Activation Kernels
// -------------------------------------------------------------------------
__global__ void relu_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : 0.0f;
    }
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
        float d_gelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
        dZ[idx] = dA[idx] * d_gelu;
    }
}

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + __expf(-Z[idx]));
        dZ[idx] = dA[idx] * sig * (1.0f - sig);
    }
}

// -------------------------------------------------------------------------
// PyTorch Extension Interface Functions
// -------------------------------------------------------------------------
torch::Tensor relu_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda() && Z.is_contiguous(), "Input must be contiguous CUDA tensor");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor relu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

torch::Tensor gelu_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda() && Z.is_contiguous(), "Input must be contiguous CUDA tensor");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_forward_kernel<<<blocks, threads>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor gelu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_backward_kernel<<<blocks, threads>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda() && Z.is_contiguous(), "Input must be contiguous CUDA tensor");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_forward_kernel<<<blocks, threads>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor sigmoid_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

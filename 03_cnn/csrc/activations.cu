#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_COEFF 0.044715f

// -------------------------------------------------------------------------
// ReLU
// -------------------------------------------------------------------------
__global__ void relu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        A[idx] = (z > 0.0f) ? z : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : 0.0f;
    }
}

// -------------------------------------------------------------------------
// GELU
// -------------------------------------------------------------------------
__global__ void gelu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
        A[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void gelu_backward_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x * x);
        float cdf = 0.5f * (1.0f + tanh_val);
        float pdf = 0.5f * x * sech2 * d_inner;
        dZ[idx] = dA[idx] * (cdf + pdf);
    }
}

// -------------------------------------------------------------------------
// Sigmoid
// -------------------------------------------------------------------------
__global__ void sigmoid_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 1.0f / (1.0f + expf(-Z[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + expf(-Z[idx]));
        dZ[idx] = dA[idx] * sig * (1.0f - sig);
    }
}

// -------------------------------------------------------------------------
// PyTorch Extension Wrappers
// -------------------------------------------------------------------------
torch::Tensor relu_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(Z.is_contiguous(), "Input must be contiguous");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_forward_kernel<<<grid, block>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor relu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dA.is_contiguous() && Z.is_contiguous(), "Inputs must be contiguous");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

torch::Tensor gelu_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(Z.is_contiguous(), "Input must be contiguous");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    gelu_forward_kernel<<<grid, block>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor gelu_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dA.is_contiguous() && Z.is_contiguous(), "Inputs must be contiguous");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    gelu_backward_kernel<<<grid, block>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor Z) {
    TORCH_CHECK(Z.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(Z.is_contiguous(), "Input must be contiguous");
    auto A = torch::empty_like(Z);
    int size = Z.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    sigmoid_forward_kernel<<<grid, block>>>(Z.data_ptr<float>(), A.data_ptr<float>(), size);
    return A;
}

torch::Tensor sigmoid_backward_cuda(torch::Tensor dA, torch::Tensor Z) {
    TORCH_CHECK(dA.is_cuda() && Z.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dA.is_contiguous() && Z.is_contiguous(), "Inputs must be contiguous");
    auto dZ = torch::empty_like(dA);
    int size = dA.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    sigmoid_backward_kernel<<<grid, block>>>(dA.data_ptr<float>(), Z.data_ptr<float>(), dZ.data_ptr<float>(), size);
    return dZ;
}

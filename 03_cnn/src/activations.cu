#include "../include/activations.cuh"
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
// LeakyReLU
// -------------------------------------------------------------------------
__global__ void leaky_relu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        A[idx] = (z > 0.0f) ? z : alpha * z;
    }
}

__global__ void leaky_relu_backward_kernel(const float* __restrict__ dA, const float* __restrict__ Z, float* __restrict__ dZ, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : alpha * dA[idx];
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
// Host Launchers
// -------------------------------------------------------------------------
void relu_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_forward_kernel<<<grid, block, 0, stream>>>(d_Z, d_A, size);
    CUDA_CHECK_LAST();
}

void relu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_backward_kernel<<<grid, block, 0, stream>>>(d_dA, d_Z, d_dZ, size);
    CUDA_CHECK_LAST();
}

void leaky_relu_forward(const float* d_Z, float* d_A, int size, float alpha, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    leaky_relu_forward_kernel<<<grid, block, 0, stream>>>(d_Z, d_A, size, alpha);
    CUDA_CHECK_LAST();
}

void leaky_relu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, float alpha, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    leaky_relu_backward_kernel<<<grid, block, 0, stream>>>(d_dA, d_Z, d_dZ, size, alpha);
    CUDA_CHECK_LAST();
}

void gelu_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    gelu_forward_kernel<<<grid, block, 0, stream>>>(d_Z, d_A, size);
    CUDA_CHECK_LAST();
}

void gelu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    gelu_backward_kernel<<<grid, block, 0, stream>>>(d_dA, d_Z, d_dZ, size);
    CUDA_CHECK_LAST();
}

void sigmoid_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    sigmoid_forward_kernel<<<grid, block, 0, stream>>>(d_Z, d_A, size);
    CUDA_CHECK_LAST();
}

void sigmoid_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream) {
    int block = 256;
    int grid = (size + block - 1) / block;
    sigmoid_backward_kernel<<<grid, block, 0, stream>>>(d_dA, d_Z, d_dZ, size);
    CUDA_CHECK_LAST();
}

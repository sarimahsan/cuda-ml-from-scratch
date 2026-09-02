#include "../include/elementwise.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"

namespace cuda_ml {
namespace kernels {

// 1. Vectorized Tensor Addition (float4)
__global__ void elementwise_add_vec4_kernel(const float* __restrict__ X,
                                            const float* __restrict__ Y,
                                            float* __restrict__ Z,
                                            int num_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 a = load_float4_aligned(X, idx);
        float4 b = load_float4_aligned(Y, idx);
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        store_float4_aligned(Z, idx, c);
    }
}

// 2. Vectorized Tensor Scale (float4)
__global__ void elementwise_scale_vec4_kernel(const float* __restrict__ X,
                                              float* __restrict__ Y,
                                              float alpha,
                                              int num_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 a = load_float4_aligned(X, idx);
        float4 c;
        c.x = alpha * a.x;
        c.y = alpha * a.y;
        c.z = alpha * a.z;
        c.w = alpha * a.w;
        store_float4_aligned(Y, idx, c);
    }
}

// 3. Fused Residual Addition (ResNet Skip Connections: Y = X + Residual)
__global__ void fused_residual_add_vec4_kernel(const float* __restrict__ X,
                                               const float* __restrict__ residual,
                                               float* __restrict__ Y,
                                               int num_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 x_val = load_float4_aligned(X, idx);
        float4 res   = load_float4_aligned(residual, idx);
        float4 out;
        out.x = x_val.x + res.x;
        out.y = x_val.y + res.y;
        out.z = x_val.z + res.z;
        out.w = x_val.w + res.w;
        store_float4_aligned(Y, idx, out);
    }
}

// 4. Bias Broadcast Add: Y = X + bias (X: M x N, bias: N)
__global__ void broadcast_bias_add_kernel(const float* __restrict__ X,
                                          const float* __restrict__ bias,
                                          float* __restrict__ Y,
                                          int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        Y[idx] = X[idx] + bias[col];
    }
}

// 5. Vectorized Elementwise Multiply (float4)
__global__ void elementwise_mul_vec4_kernel(const float* __restrict__ X,
                                            const float* __restrict__ Y,
                                            float* __restrict__ Z,
                                            int num_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 a = load_float4_aligned(X, idx);
        float4 b = load_float4_aligned(Y, idx);
        float4 c;
        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;
        store_float4_aligned(Z, idx, c);
    }
}

// Host Launchers
void elementwise_add(const float* X, const float* Y, float* Z, int N, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        elementwise_add_vec4_kernel<<<blocks, threads, 0, stream>>>(X, Y, Z, num_vec4);
    }
}

void elementwise_scale(const float* X, float* Y, float alpha, int N, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        elementwise_scale_vec4_kernel<<<blocks, threads, 0, stream>>>(X, Y, alpha, num_vec4);
    }
}

void fused_residual_add(const float* X, const float* residual, float* Y, int N, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        fused_residual_add_vec4_kernel<<<blocks, threads, 0, stream>>>(X, residual, Y, num_vec4);
    }
}

void broadcast_bias_add(const float* X, const float* bias, float* Y, int M, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    broadcast_bias_add_kernel<<<grid, block, 0, stream>>>(X, bias, Y, M, N);
}

void elementwise_mul(const float* X, const float* Y, float* Z, int N, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        elementwise_mul_vec4_kernel<<<blocks, threads, 0, stream>>>(X, Y, Z, num_vec4);
    }
}

} // namespace kernels
} // namespace cuda_ml

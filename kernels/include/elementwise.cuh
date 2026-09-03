#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// Vectorized Elementwise & Tensor Operations (float4 saturation)
// ============================================================================

// 1. Vectorized Tensor Add: Z = X + Y (in-place or out-of-place)
void elementwise_add(const float* X, const float* Y, float* Z, int N, cudaStream_t stream = 0);

// 2. Vectorized Tensor Scale: Y = alpha * X
void elementwise_scale(const float* X, float* Y, float alpha, int N, cudaStream_t stream = 0);

// 3. Fused Residual Addition: Y = X + Residual (e.g. ResNet skip connection)
void fused_residual_add(const float* X, const float* residual, float* Y, int N, cudaStream_t stream = 0);

// 4. Bias Addition with 1D broadcasting: Y = X + bias (X is M x N, bias is N)
void broadcast_bias_add(const float* X, const float* bias, float* Y, int M, int N, cudaStream_t stream = 0);

// 5. Vectorized Elementwise Multiply (Hadamard): Z = X * Y
void elementwise_mul(const float* X, const float* Y, float* Z, int N, cudaStream_t stream = 0);

// 6. Fast GPU Dropout Forward & Backward with Philox-4x32 PRNG
void dropout_forward(const float* X, float* Y, uint8_t* mask, int N, float p,
                     uint64_t seed, uint64_t offset = 0, cudaStream_t stream = 0);

void dropout_backward(const float* dY, const uint8_t* mask, float* dX, int N, float p,
                      cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

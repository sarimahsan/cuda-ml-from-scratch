#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// Normalization Engine: LayerNorm, RMSNorm, BatchNorm (1-Pass Welford Algorithm)
// ============================================================================

// 1. Layer Normalization Forward: y = (x - mean) / sqrt(var + eps) * gamma + beta
// X: (N x D), gamma: (D), beta: (D), Y: (N x D)
// mean: (N), rstd: (N) [saved for backward]
void layernorm_forward(const float* X, const float* gamma, const float* beta,
                       float* Y, float* mean, float* rstd,
                       int N, int D, float eps = 1e-5f, cudaStream_t stream = 0);

// Layer Normalization Backward: computes dX, dgamma, dbeta
void layernorm_backward(const float* dY, const float* X, const float* gamma,
                        const float* mean, const float* rstd,
                        float* dX, float* dgamma, float* dbeta,
                        int N, int D, cudaStream_t stream = 0);

// 2. RMS Normalization (Root Mean Square Layer Normalization - used in LLaMA / Transformers)
// y = x / sqrt(mean(x^2) + eps) * gamma
void rmsnorm_forward(const float* X, const float* gamma, float* Y, float* rstd,
                     int N, int D, float eps = 1e-5f, cudaStream_t stream = 0);

void rmsnorm_backward(const float* dY, const float* X, const float* gamma, const float* rstd,
                      float* dX, float* dgamma,
                      int N, int D, cudaStream_t stream = 0);

// 3. Batch Normalization (2D Spatial: N x C x H x W)
void batchnorm2d_forward(const float* X, const float* gamma, const float* beta,
                         float* running_mean, float* running_var,
                         float* Y, float* save_mean, float* save_rstd,
                         int N, int C, int H, int W,
                         bool training, float momentum = 0.1f, float eps = 1e-5f,
                         cudaStream_t stream = 0);

void batchnorm2d_backward(const float* dY, const float* X, const float* gamma,
                          const float* save_mean, const float* save_rstd,
                          float* dX, float* dgamma, float* dbeta,
                          int N, int C, int H, int W, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// Vectorized GPU Optimizers (Adam, AdamW, SGD with Momentum)
// ============================================================================

// 1. SGD with Momentum: v = momentum * v + grad; param -= lr * v
void sgd_momentum_step(float* param, float* velocity, const float* grad,
                       float lr, float momentum, int size, cudaStream_t stream = 0);

// 2. Adam In-Place Step
void adam_step(float* param, float* m, float* v, const float* grad,
               float lr, float beta1, float beta2, float eps,
               int step, int size, float weight_decay = 0.0f,
               cudaStream_t stream = 0);

// 3. Vectorized Gradient Clipping by L2 Norm
void clip_grad_norm(float* grad, int size, float max_norm, float actual_norm,
                    cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Vectorized GPU Optimizers and Gradient Clipping
// -----------------------------------------------------------------------------

// In-place SGD with Momentum: v = beta * v + g; w = w - lr * v
void launch_sgd_momentum_step(
    float* d_weights,
    float* d_v,
    const float* d_grads,
    int size,
    float lr,
    float momentum,
    cudaStream_t stream = 0
);

// In-place Adam Optimizer Step:
// m = beta1 * m + (1 - beta1) * g
// v = beta2 * v + (1 - beta2) * g^2
// w = w - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
void launch_adam_step(
    float* d_weights,
    float* d_m,
    float* d_v,
    const float* d_grads,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    cudaStream_t stream = 0
);

// Global Gradient Norm Calculation and In-Place Gradient Clipping
// Scales all gradients if sum(norm^2) > max_norm^2
void launch_clip_grad_norm(
    float** d_grad_ptrs,
    const int* grad_sizes,
    int num_tensors,
    float max_norm,
    cudaStream_t stream = 0
);

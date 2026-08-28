#pragma once

#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// Optimizer Kernel Declarations
// -------------------------------------------------------------------------

// In-place SGD with Momentum: v = beta * v + grad; param = param - lr * v
void launch_sgd_momentum(
    float* d_param,
    float* d_velocity,
    const float* d_grad,
    float lr,
    float momentum,
    int size,
    cudaStream_t stream = 0
);

// In-place Adam Optimizer Step
void launch_adam(
    float* d_param,
    float* d_m,
    float* d_v,
    const float* d_grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    int size,
    cudaStream_t stream = 0
);

#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -------------------------------------------------------------------------
// Linear (GEMM) CUDA Kernel Declarations
// -------------------------------------------------------------------------

// Forward Linear: Z = X * W + b
// X: [M x K], W: [K x N], b: [N], Z: [M x N]
void launch_linear_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_Z,
    int M, int K, int N,
    cudaStream_t stream = 0
);

// Backward Linear Gradients:
// dW = X^T * dZ   [K x N]
// db = sum_rows(dZ) [N]
// dX = dZ * W^T   [M x K] (optional)
void launch_linear_backward(
    const float* d_dZ,
    const float* d_X,
    const float* d_W,
    float* d_dW,
    float* d_db,
    float* d_dX,
    int M, int K, int N,
    bool compute_dX = true,
    cudaStream_t stream = 0
);

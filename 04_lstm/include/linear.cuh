#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Linear / GEMM Layer declarations:
// Forward:  Y = X * W^T + b  or  Y = X * W + b
// Backward: dW = dY^T * X,   db = sum_rows(dY),  dX = dY * W
// -----------------------------------------------------------------------------

// Tiled GEMM with bias addition: Out [M x N] = A [M x K] * B [K x N] + Bias [N] (if bias != nullptr)
void launch_gemm_forward(
    const float* d_A,
    const float* d_B,
    const float* d_bias,
    float* d_Out,
    int M,
    int K,
    int N,
    cudaStream_t stream = 0
);

// Gradient with respect to weights: dW [K x N] += A^T [K x M] * dY [M x N]
void launch_gemm_backward_weights(
    const float* d_A,   // [M x K]
    const float* d_dY,  // [M x N]
    float* d_dW,        // [K x N]
    int M,
    int K,
    int N,
    bool accumulate = false,
    cudaStream_t stream = 0
);

// Gradient with respect to inputs: dX [M x K] = dY [M x N] * W^T [N x K]
void launch_gemm_backward_data(
    const float* d_dY,  // [M x N]
    const float* d_W,   // [K x N]
    float* d_dX,        // [M x K]
    int M,
    int K,
    int N,
    cudaStream_t stream = 0
);

// Gradient with respect to bias: db [N] += sum_rows(dY [M x N])
void launch_gemm_backward_bias(
    const float* d_dY,
    float* d_db,
    int M,
    int N,
    bool accumulate = false,
    cudaStream_t stream = 0
);

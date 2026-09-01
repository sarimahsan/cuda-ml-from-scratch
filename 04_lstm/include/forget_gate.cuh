#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Forget Gate: f_t = sigmoid(Z_f) = 1 / (1 + exp(-Z_f))
// Controls what proportion of previous cell state c_{t-1} is retained.
// -----------------------------------------------------------------------------

// Forward: computes f_t = sigmoid(Z_f)
void launch_forget_gate_forward(
    const float* d_Z_f,
    float* d_f,
    int size,
    cudaStream_t stream = 0
);

// Backward: computes dZ_f = df_t * f_t * (1 - f_t)
void launch_forget_gate_backward(
    const float* d_df,
    const float* d_f,
    float* d_dZ_f,
    int size,
    cudaStream_t stream = 0
);

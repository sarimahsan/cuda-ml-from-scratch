#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Input Gate: i_t = sigmoid(Z_i) = 1 / (1 + exp(-Z_i))
// Controls what proportion of new candidate information enters the cell state.
// -----------------------------------------------------------------------------

// Forward: computes i_t = sigmoid(Z_i)
void launch_input_gate_forward(
    const float* d_Z_i,
    float* d_i,
    int size,
    cudaStream_t stream = 0
);

// Backward: computes dZ_i = di_t * i_t * (1 - i_t)
void launch_input_gate_backward(
    const float* d_di,
    const float* d_i,
    float* d_dZ_i,
    int size,
    cudaStream_t stream = 0
);

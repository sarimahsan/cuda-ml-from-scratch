#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Cell Candidate Gate (Modulation): g_t = tanh(Z_g) = (exp(Z_g) - exp(-Z_g)) / (exp(Z_g) + exp(-Z_g))
// Computes new candidate values that could be added to the cell state.
// -----------------------------------------------------------------------------

// Forward: computes g_t = tanh(Z_g)
void launch_candidate_gate_forward(
    const float* d_Z_g,
    float* d_g,
    int size,
    cudaStream_t stream = 0
);

// Backward: computes dZ_g = dg_t * (1 - g_t^2)
void launch_candidate_gate_backward(
    const float* d_dg,
    const float* d_g,
    float* d_dZ_g,
    int size,
    cudaStream_t stream = 0
);

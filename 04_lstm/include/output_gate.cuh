#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Output Gate: o_t = sigmoid(Z_o) = 1 / (1 + exp(-Z_o))
// Controls what parts of the updated cell state c_t are emitted to hidden state h_t.
// -----------------------------------------------------------------------------

// Forward: computes o_t = sigmoid(Z_o)
void launch_output_gate_forward(
    const float* d_Z_o,
    float* d_o,
    int size,
    cudaStream_t stream = 0
);

// Backward: computes dZ_o = do_t * o_t * (1 - o_t)
void launch_output_gate_backward(
    const float* d_do,
    const float* d_o,
    float* d_dZ_o,
    int size,
    cudaStream_t stream = 0
);

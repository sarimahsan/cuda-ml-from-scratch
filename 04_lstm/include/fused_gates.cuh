#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Fused 4-Gate LSTM Elementwise Kernel:
// Given pre-activations Z = [Z_i, Z_f, Z_g, Z_o] of shape [N x 4H],
// computes all 4 gates (i, f, g, o), updated cell state c_t, tanh(c_t), and hidden state h_t
// in a single fused kernel to minimize global memory roundtrips.
// -----------------------------------------------------------------------------

void launch_fused_lstm_gates_forward(
    const float* d_gates_preact, // [N x 4H]
    const float* d_c_prev,       // [N x H]
    float* d_gates_act,          // [N x 4H] -> [i, f, g, o]
    float* d_c_next,             // [N x H]
    float* d_tanh_c,             // [N x H]
    float* d_h_next,             // [N x H]
    int N,
    int H,
    cudaStream_t stream = 0
);

void launch_fused_lstm_gates_backward(
    const float* d_dh,           // [N x H]
    const float* d_dc_next,      // [N x H]
    const float* d_gates_act,    // [N x 4H] -> [i, f, g, o]
    const float* d_c_prev,       // [N x H]
    const float* d_c_next,       // [N x H]
    const float* d_tanh_c,       // [N x H]
    float* d_dgates_preact,      // [N x 4H] -> [dZ_i, dZ_f, dZ_g, dZ_o]
    float* d_dc_prev,            // [N x H]
    int N,
    int H,
    cudaStream_t stream = 0
);

#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Cell State & Hidden State Update:
// Forward:
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
// Backward:
//   dc_t_total = dh_t * o_t * (1 - tanh^2(c_t)) + dc_{t+1}
//   dc_{t-1} = dc_t_total * f_t
//   df_t = dc_t_total * c_{t-1}
//   di_t = dc_t_total * g_t
//   dg_t = dc_t_total * i_t
//   do_t = dh_t * tanh(c_t)
// -----------------------------------------------------------------------------

// Forward cell state & hidden state
void launch_cell_state_forward(
    const float* d_f,
    const float* d_c_prev,
    const float* d_i,
    const float* d_g,
    const float* d_o,
    float* d_c,
    float* d_tanh_c,
    float* d_h,
    int size,
    cudaStream_t stream = 0
);

// Backward gradients for gates from cell state & hidden state
void launch_cell_state_backward(
    const float* d_dh,
    const float* d_dc_next,
    const float* d_o,
    const float* d_tanh_c,
    const float* d_c_prev,
    const float* d_f,
    const float* d_i,
    const float* d_g,
    float* d_dc_prev,
    float* d_df,
    float* d_di,
    float* d_dg,
    float* d_do,
    int size,
    cudaStream_t stream = 0
);

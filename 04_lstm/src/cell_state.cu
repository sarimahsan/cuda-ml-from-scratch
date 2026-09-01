#include "../include/cell_state.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Cell State & Hidden State Update Kernels:
// Forward:
//   c_t = f_t * c_{t-1} + i_t * g_t
//   tanh_c = tanh(c_t)
//   h_t = o_t * tanh_c
// Backward:
//   dc_t_total = dh_t * o_t * (1 - tanh_c^2) + dc_{t+1}
//   dc_{t-1} = dc_t_total * f_t
//   df_t = dc_t_total * c_{t-1}
//   di_t = dc_t_total * g_t
//   dg_t = dc_t_total * i_t
//   do_t = dh_t * tanh_c
// -----------------------------------------------------------------------------

__global__ void cell_state_forward_kernel(
    const float* __restrict__ d_f,
    const float* __restrict__ d_c_prev,
    const float* __restrict__ d_i,
    const float* __restrict__ d_g,
    const float* __restrict__ d_o,
    float* __restrict__ d_c,
    float* __restrict__ d_tanh_c,
    float* __restrict__ d_h,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f_val = d_f[idx];
        float c_prev_val = d_c_prev ? d_c_prev[idx] : 0.0f;
        float i_val = d_i[idx];
        float g_val = d_g[idx];
        float o_val = d_o[idx];

        // c_t = f_t * c_{t-1} + i_t * g_t
        float c_val = f_val * c_prev_val + i_val * g_val;
        float tc_val = tanhf(c_val);
        // h_t = o_t * tanh(c_t)
        float h_val = o_val * tc_val;

        d_c[idx] = c_val;
        if (d_tanh_c) d_tanh_c[idx] = tc_val;
        d_h[idx] = h_val;
    }
}

__global__ void cell_state_backward_kernel(
    const float* __restrict__ d_dh,
    const float* __restrict__ d_dc_next,
    const float* __restrict__ d_o,
    const float* __restrict__ d_tanh_c,
    const float* __restrict__ d_c_prev,
    const float* __restrict__ d_f,
    const float* __restrict__ d_i,
    const float* __restrict__ d_g,
    float* __restrict__ d_dc_prev,
    float* __restrict__ d_df,
    float* __restrict__ d_di,
    float* __restrict__ d_dg,
    float* __restrict__ d_do,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dh = d_dh[idx];
        float dc_next = d_dc_next ? d_dc_next[idx] : 0.0f;
        float o_val = d_o[idx];
        float tc_val = d_tanh_c[idx];
        float c_prev_val = d_c_prev ? d_c_prev[idx] : 0.0f;
        float f_val = d_f[idx];
        float i_val = d_i[idx];
        float g_val = d_g[idx];

        // 1. do_t = dh_t * tanh(c_t)
        float do_val = dh * tc_val;

        // 2. dc_t_total = dh_t * o_t * (1 - tanh_c^2) + dc_next
        float dtanh_c = dh * o_val;
        float dc_total = dtanh_c * (1.0f - tc_val * tc_val) + dc_next;

        // 3. dc_prev = dc_total * f_t
        float dc_prev_val = dc_total * f_val;

        // 4. df_t = dc_total * c_{t-1}
        float df_val = dc_total * c_prev_val;

        // 5. di_t = dc_total * g_t
        float di_val = dc_total * g_val;

        // 6. dg_t = dc_total * i_t
        float dg_val = dc_total * i_val;

        if (d_do) d_do[idx] = do_val;
        if (d_dc_prev) d_dc_prev[idx] = dc_prev_val;
        if (d_df) d_df[idx] = df_val;
        if (d_di) d_di[idx] = di_val;
        if (d_dg) d_dg[idx] = dg_val;
    }
}

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
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cell_state_forward_kernel<<<blocks, threads, 0, stream>>>(
        d_f, d_c_prev, d_i, d_g, d_o, d_c, d_tanh_c, d_h, size
    );
    CUDA_KERNEL_CHECK();
}

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
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cell_state_backward_kernel<<<blocks, threads, 0, stream>>>(
        d_dh, d_dc_next, d_o, d_tanh_c, d_c_prev, d_f, d_i, d_g,
        d_dc_prev, d_df, d_di, d_dg, d_do, size
    );
    CUDA_KERNEL_CHECK();
}

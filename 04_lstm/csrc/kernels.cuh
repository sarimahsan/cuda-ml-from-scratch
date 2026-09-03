#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// High-Performance Fused LSTM Single-Pass Step Forward Kernel
// Fuses: (G_ih + G_hh + b_hh) -> 4-gate activations -> cell state -> hidden state
// -----------------------------------------------------------------------------

__global__ inline void fused_lstm_step_forward_kernel_torch(
    const float* __restrict__ d_g_ih,
    const float* __restrict__ d_g_hh,
    const float* __restrict__ d_b_hh,
    const float* __restrict__ d_c_prev,
    float* __restrict__ d_g_act,
    float* __restrict__ d_c_next,
    float* __restrict__ d_tanh_c,
    float* __restrict__ d_h_next,
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_4h = n * (4 * H);

        float z_i = d_g_ih[base_4h + h] + d_g_hh[base_4h + h];
        float z_f = d_g_ih[base_4h + H + h] + d_g_hh[base_4h + H + h];
        float z_g = d_g_ih[base_4h + 2 * H + h] + d_g_hh[base_4h + 2 * H + h];
        float z_o = d_g_ih[base_4h + 3 * H + h] + d_g_hh[base_4h + 3 * H + h];

        if (d_b_hh != nullptr) {
            z_i += d_b_hh[h];
            z_f += d_b_hh[H + h];
            z_g += d_b_hh[2 * H + h];
            z_o += d_b_hh[3 * H + h];
        }

        float i_val = sigmoidf_device(z_i);
        float f_val = sigmoidf_device(z_f);
        float g_val = tanhf(z_g);
        float o_val = sigmoidf_device(z_o);

        if (d_g_act != nullptr) {
            d_g_act[base_4h + h] = i_val;
            d_g_act[base_4h + H + h] = f_val;
            d_g_act[base_4h + 2 * H + h] = g_val;
            d_g_act[base_4h + 3 * H + h] = o_val;
        }

        float c_prev = (d_c_prev != nullptr) ? d_c_prev[idx] : 0.0f;
        float c_next = f_val * c_prev + i_val * g_val;
        float tc = tanhf(c_next);
        float h_next = o_val * tc;

        d_c_next[idx] = c_next;
        if (d_tanh_c != nullptr) d_tanh_c[idx] = tc;
        d_h_next[idx] = h_next;
    }
}

// -----------------------------------------------------------------------------
// Standalone Fused 4-Gate Forward & Backward (with fast math)
// -----------------------------------------------------------------------------

__global__ inline void fused_lstm_gates_forward_kernel_torch(
    const float* __restrict__ d_gates_preact,
    const float* __restrict__ d_c_prev,
    float* __restrict__ d_gates_act,
    float* __restrict__ d_c_next,
    float* __restrict__ d_tanh_c,
    float* __restrict__ d_h_next,
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_4h = n * (4 * H);
        float z_i = d_gates_preact[base_4h + h];
        float z_f = d_gates_preact[base_4h + H + h];
        float z_g = d_gates_preact[base_4h + 2 * H + h];
        float z_o = d_gates_preact[base_4h + 3 * H + h];

        float i_val = sigmoidf_device(z_i);
        float f_val = sigmoidf_device(z_f);
        float g_val = tanhf(z_g);
        float o_val = sigmoidf_device(z_o);

        if (d_gates_act) {
            d_gates_act[base_4h + h] = i_val;
            d_gates_act[base_4h + H + h] = f_val;
            d_gates_act[base_4h + 2 * H + h] = g_val;
            d_gates_act[base_4h + 3 * H + h] = o_val;
        }

        float c_prev = d_c_prev ? d_c_prev[idx] : 0.0f;
        float c_next = f_val * c_prev + i_val * g_val;
        float tc = tanhf(c_next);
        float h_next = o_val * tc;

        d_c_next[idx] = c_next;
        if (d_tanh_c) d_tanh_c[idx] = tc;
        d_h_next[idx] = h_next;
    }
}

__global__ inline void fused_lstm_gates_backward_kernel_torch(
    const float* __restrict__ d_dh,
    const float* __restrict__ d_dc_next,
    const float* __restrict__ d_gates_act,
    const float* __restrict__ d_c_prev,
    const float* __restrict__ d_c_next,
    const float* __restrict__ d_tanh_c,
    float* __restrict__ d_dgates_preact,
    float* __restrict__ d_dc_prev,
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_4h = n * (4 * H);
        float i_val = d_gates_act[base_4h + h];
        float f_val = d_gates_act[base_4h + H + h];
        float g_val = d_gates_act[base_4h + 2 * H + h];
        float o_val = d_gates_act[base_4h + 3 * H + h];

        float dh = d_dh[idx];
        float dc_next = d_dc_next ? d_dc_next[idx] : 0.0f;
        float tc = d_tanh_c ? d_tanh_c[idx] : tanhf(d_c_next[idx]);
        float c_prev = d_c_prev ? d_c_prev[idx] : 0.0f;

        float do_val = dh * tc;
        float dz_o = do_val * o_val * (1.0f - o_val);

        float dtanh_c = dh * o_val;
        float dc_total = dtanh_c * (1.0f - tc * tc) + dc_next;

        float dc_prev_val = dc_total * f_val;
        if (d_dc_prev) d_dc_prev[idx] = dc_prev_val;

        float df_val = dc_total * c_prev;
        float dz_f = df_val * f_val * (1.0f - f_val);

        float di_val = dc_total * g_val;
        float dz_i = di_val * i_val * (1.0f - i_val);

        float dg_val = dc_total * i_val;
        float dz_g = dg_val * (1.0f - g_val * g_val);

        d_dgates_preact[base_4h + h] = dz_i;
        d_dgates_preact[base_4h + H + h] = dz_f;
        d_dgates_preact[base_4h + 2 * H + h] = dz_g;
        d_dgates_preact[base_4h + 3 * H + h] = dz_o;
    }
}

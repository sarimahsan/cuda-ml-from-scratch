#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// Shared 2D Tiled GEMM Kernels (Fallback / Standalone)
// -----------------------------------------------------------------------------

__global__ inline void gemm_forward_kernel_torch(
    const float* __restrict__ d_A,
    const float* __restrict__ d_B,
    const float* __restrict__ d_bias,
    float* __restrict__ d_Out,
    int M,
    int K,
    int N
) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = d_A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = d_B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (d_bias != nullptr) {
            sum += d_bias[col];
        }
        d_Out[row * N + col] = sum;
    }
}

__global__ inline void gemm_backward_weights_kernel_torch(
    const float* __restrict__ d_A,
    const float* __restrict__ d_dY,
    float* __restrict__ d_dW,
    int M,
    int K,
    int N,
    bool accumulate = false
) {
    __shared__ float s_AT[TILE_DIM][TILE_DIM];
    __shared__ float s_dY[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (M + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_row = t * TILE_DIM + threadIdx.x;
        if (row < K && a_row < M) {
            s_AT[threadIdx.y][threadIdx.x] = d_A[a_row * K + row];
        } else {
            s_AT[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int dy_row = t * TILE_DIM + threadIdx.y;
        if (dy_row < M && col < N) {
            s_dY[threadIdx.y][threadIdx.x] = d_dY[dy_row * N + col];
        } else {
            s_dY[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_AT[threadIdx.y][k] * s_dY[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        if (accumulate) {
            d_dW[row * N + col] += sum;
        } else {
            d_dW[row * N + col] = sum;
        }
    }
}

__global__ inline void gemm_backward_data_kernel_torch(
    const float* __restrict__ d_dY,
    const float* __restrict__ d_W,
    float* __restrict__ d_dX,
    int M,
    int K,
    int N
) {
    __shared__ float s_dY[TILE_DIM][TILE_DIM];
    __shared__ float s_WT[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int dy_col = t * TILE_DIM + threadIdx.x;
        if (row < M && dy_col < N) {
            s_dY[threadIdx.y][threadIdx.x] = d_dY[row * N + dy_col];
        } else {
            s_dY[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int wt_row = t * TILE_DIM + threadIdx.y;
        if (wt_row < N && col < K) {
            s_WT[threadIdx.y][threadIdx.x] = d_W[col * N + wt_row];
        } else {
            s_WT[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_dY[threadIdx.y][k] * s_WT[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        d_dX[row * K + col] = sum;
    }
}

__global__ inline void gemm_backward_bias_kernel_torch(
    const float* __restrict__ d_dY,
    float* __restrict__ d_db,
    int M,
    int N,
    bool accumulate = false
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += d_dY[row * N + col];
        }
        if (accumulate) {
            d_db[col] += sum;
        } else {
            d_db[col] = sum;
        }
    }
}

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

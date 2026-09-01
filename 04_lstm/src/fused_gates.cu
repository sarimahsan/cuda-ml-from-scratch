#include "../include/fused_gates.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Fused 4-Gate LSTM Elementwise Kernel:
// Given pre-activations Z = [Z_i, Z_f, Z_g, Z_o] of shape [N x 4H],
// computes all 4 gates (i, f, g, o), updated cell state c_t, tanh(c_t), and hidden state h_t
// in a single fused kernel to minimize global memory roundtrips.
// Layout in memory:
// Row n:
//   i: [0 .. H-1]
//   f: [H .. 2H-1]
//   g: [2H .. 3H-1]
//   o: [3H .. 4H-1]
// -----------------------------------------------------------------------------

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void fused_lstm_gates_forward_kernel(
    const float* __restrict__ d_gates_preact, // [N x 4H]
    const float* __restrict__ d_c_prev,       // [N x H]
    float* __restrict__ d_gates_act,          // [N x 4H] -> [i, f, g, o]
    float* __restrict__ d_c_next,             // [N x H]
    float* __restrict__ d_tanh_c,             // [N x H]
    float* __restrict__ d_h_next,             // [N x H]
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

        float i_val = sigmoidf_dev(z_i);
        float f_val = sigmoidf_dev(z_f);
        float g_val = tanhf(z_g);
        float o_val = sigmoidf_dev(z_o);

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

__global__ void fused_lstm_gates_backward_kernel(
    const float* __restrict__ d_dh,           // [N x H]
    const float* __restrict__ d_dc_next,      // [N x H]
    const float* __restrict__ d_gates_act,    // [N x 4H] -> [i, f, g, o]
    const float* __restrict__ d_c_prev,       // [N x H]
    const float* __restrict__ d_c_next,       // [N x H]
    const float* __restrict__ d_tanh_c,       // [N x H]
    float* __restrict__ d_dgates_preact,      // [N x 4H] -> [dZ_i, dZ_f, dZ_g, dZ_o]
    float* __restrict__ d_dc_prev,            // [N x H]
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

        // Gradient of output gate
        float do_val = dh * tc;
        float dz_o = do_val * o_val * (1.0f - o_val);

        // Gradient of cell state
        float dtanh_c = dh * o_val;
        float dc_total = dtanh_c * (1.0f - tc * tc) + dc_next;

        // Gradient of previous cell state
        float dc_prev_val = dc_total * f_val;
        if (d_dc_prev) d_dc_prev[idx] = dc_prev_val;

        // Gradient of forget gate
        float df_val = dc_total * c_prev;
        float dz_f = df_val * f_val * (1.0f - f_val);

        // Gradient of input gate
        float di_val = dc_total * g_val;
        float dz_i = di_val * i_val * (1.0f - i_val);

        // Gradient of candidate gate
        float dg_val = dc_total * i_val;
        float dz_g = dg_val * (1.0f - g_val * g_val);

        // Store preactivation gradients
        d_dgates_preact[base_4h + h] = dz_i;
        d_dgates_preact[base_4h + H + h] = dz_f;
        d_dgates_preact[base_4h + 2 * H + h] = dz_g;
        d_dgates_preact[base_4h + 3 * H + h] = dz_o;
    }
}

void launch_fused_lstm_gates_forward(
    const float* d_gates_preact,
    const float* d_c_prev,
    float* d_gates_act,
    float* d_c_next,
    float* d_tanh_c,
    float* d_h_next,
    int N,
    int H,
    cudaStream_t stream
) {
    int total = N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_lstm_gates_forward_kernel<<<blocks, threads, 0, stream>>>(
        d_gates_preact, d_c_prev, d_gates_act, d_c_next, d_tanh_c, d_h_next, N, H
    );
    CUDA_KERNEL_CHECK();
}

void launch_fused_lstm_gates_backward(
    const float* d_dh,
    const float* d_dc_next,
    const float* d_gates_act,
    const float* d_c_prev,
    const float* d_c_next,
    const float* d_tanh_c,
    float* d_dgates_preact,
    float* d_dc_prev,
    int N,
    int H,
    cudaStream_t stream
) {
    int total = N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_lstm_gates_backward_kernel<<<blocks, threads, 0, stream>>>(
        d_dh, d_dc_next, d_gates_act, d_c_prev, d_c_next, d_tanh_c, d_dgates_preact, d_dc_prev, N, H
    );
    CUDA_KERNEL_CHECK();
}

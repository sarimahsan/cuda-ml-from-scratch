#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ml {
namespace gru {

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// -----------------------------------------------------------------------------
// 1. In-place Gradient Accumulation Kernel: Accum += Delta
// -----------------------------------------------------------------------------
__global__ void accumulate_inplace_kernel(float* __restrict__ accum,
                                         const float* __restrict__ delta,
                                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        accum[idx] += delta[idx];
    }
}

// -----------------------------------------------------------------------------
// 2. Fused GRU Step Forward Kernel
// Slices [0:H] = Reset (r), [H:2H] = Update (z), [2H:3H] = Candidate (n)
// -----------------------------------------------------------------------------
__global__ void gru_step_forward_kernel(
    const float* __restrict__ g_ih,     // [N, 3H]
    const float* __restrict__ g_hh,     // [N, 3H]
    const float* __restrict__ b_hh,     // [3H] (optional)
    const float* __restrict__ h_prev,   // [N, H]
    float* __restrict__ gates_act,      // [N, 3H] -> [r, z, n]
    float* __restrict__ h_out,          // [N, H]
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_3h = n * (3 * H);

        // Pre-activation inputs
        float g_ih_r = g_ih[base_3h + h];
        float g_ih_z = g_ih[base_3h + H + h];
        float g_ih_n = g_ih[base_3h + 2 * H + h];

        float g_hh_r = g_hh[base_3h + h];
        float g_hh_z = g_hh[base_3h + H + h];
        float g_hh_n = g_hh[base_3h + 2 * H + h];

        if (b_hh != nullptr) {
            g_hh_r += b_hh[h];
            g_hh_z += b_hh[H + h];
            g_hh_n += b_hh[2 * H + h];
        }

        // 1. Reset & Update gates
        float r_val = sigmoidf_device(g_ih_r + g_hh_r);
        float z_val = sigmoidf_device(g_ih_z + g_hh_z);

        // 2. Candidate hidden state
        float n_val = tanhf(g_ih_n + r_val * g_hh_n);

        // 3. Hidden state output
        float hp = (h_prev != nullptr) ? h_prev[idx] : 0.0f;
        float h_new = (1.0f - z_val) * n_val + z_val * hp;

        // Save gate activations for backward
        if (gates_act != nullptr) {
            gates_act[base_3h + h] = r_val;
            gates_act[base_3h + H + h] = z_val;
            gates_act[base_3h + 2 * H + h] = n_val;
        }

        h_out[idx] = h_new;
    }
}

// -----------------------------------------------------------------------------
// 3. Fused GRU Step Backward Kernel (with Fused In-Register dh Accumulation)
// -----------------------------------------------------------------------------
__global__ void gru_step_backward_kernel(
    const float* __restrict__ dh_incoming,  // [N, H] -> dH_seq[t]
    const float* __restrict__ dh_recurrent, // [N, H] -> dh_next from step t+1
    const float* __restrict__ h_prev,       // [N, H]
    const float* __restrict__ gates_act,    // [N, 3H] -> [r, z, n]
    const float* __restrict__ g_hh,         // [N, 3H] -> raw recurrent projections
    const float* __restrict__ b_hh,         // [3H] (optional)
    float* __restrict__ dg_ih,              // [N, 3H] -> [dz_r, dz_z, dz_n]
    float* __restrict__ dg_hh,              // [N, 3H] -> [dz_r, dz_z, dG_hh_n]
    float* __restrict__ dh_prev_direct,     // [N, H] -> direct gradient contribution (dh * z)
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_3h = n * (3 * H);

        // Fused in-register accumulation: dh = dh_incoming + dh_recurrent
        float dh = (dh_incoming != nullptr ? dh_incoming[idx] : 0.0f) + 
                   (dh_recurrent != nullptr ? dh_recurrent[idx] : 0.0f);
        float hp = (h_prev != nullptr) ? h_prev[idx] : 0.0f;

        float r_val = gates_act[base_3h + h];
        float z_val = gates_act[base_3h + H + h];
        float n_val = gates_act[base_3h + 2 * H + h];

        float g_hh_n_val = g_hh[base_3h + 2 * H + h];
        if (b_hh != nullptr) {
            g_hh_n_val += b_hh[2 * H + h];
        }

        // Gradients of output: h_t = (1 - z) * n + z * h_prev
        float dn = dh * (1.0f - z_val);
        float dz = dh * (hp - n_val);

        // Pre-activations
        float dz_z = dz * z_val * (1.0f - z_val);
        float dz_n = dn * (1.0f - n_val * n_val);

        float dr = dz_n * g_hh_n_val;
        float dz_r = dr * r_val * (1.0f - r_val);

        float dg_hh_n = dz_n * r_val;

        // Save dg_ih: [dz_r, dz_z, dz_n]
        dg_ih[base_3h + h] = dz_r;
        dg_ih[base_3h + H + h] = dz_z;
        dg_ih[base_3h + 2 * H + h] = dz_n;

        // Save dg_hh: [dz_r, dz_z, dg_hh_n]
        dg_hh[base_3h + h] = dz_r;
        dg_hh[base_3h + H + h] = dz_z;
        dg_hh[base_3h + 2 * H + h] = dg_hh_n;

        // Direct path to previous hidden state: dh_prev += dh * z
        if (dh_prev_direct != nullptr) {
            dh_prev_direct[idx] = dh * z_val;
        }
    }
}

// -----------------------------------------------------------------------------
// Host Launchers
// -----------------------------------------------------------------------------
void launch_gru_step_forward(
    const float* g_ih,
    const float* g_hh,
    const float* b_hh,
    const float* h_prev,
    float* gates_act,
    float* h_out,
    int N,
    int H,
    cudaStream_t stream
) {
    int total_elements = N * H;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gru_step_forward_kernel<<<blocks, threads, 0, stream>>>(
        g_ih, g_hh, b_hh, h_prev, gates_act, h_out, N, H);
}

void launch_gru_step_backward(
    const float* dh_incoming,
    const float* dh_recurrent,
    const float* h_prev,
    const float* gates_act,
    const float* g_hh,
    const float* b_hh,
    float* dg_ih,
    float* dg_hh,
    float* dh_prev_direct,
    int N,
    int H,
    cudaStream_t stream
) {
    int total_elements = N * H;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gru_step_backward_kernel<<<blocks, threads, 0, stream>>>(
        dh_incoming, dh_recurrent, h_prev, gates_act, g_hh, b_hh, dg_ih, dg_hh, dh_prev_direct, N, H);
}

} // namespace gru
} // namespace cuda_ml

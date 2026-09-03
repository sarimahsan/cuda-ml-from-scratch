#include <cuda_runtime.h>
#include <cmath>

namespace cuda_ml {
namespace rnn {

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
// 2. Fused RNN Step Forward Kernel: h_t = tanh(g_ih + g_hh + b_hh) or relu
// -----------------------------------------------------------------------------
__global__ void rnn_step_forward_kernel(const float* __restrict__ g_ih,
                                        const float* __restrict__ g_hh,
                                        const float* __restrict__ b_hh,
                                        float* __restrict__ h_out,
                                        int size,
                                        int H,
                                        int activation_type) { // 0 = tanh, 1 = relu
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int h = idx % H;
        float z = g_ih[idx] + g_hh[idx];
        if (b_hh != nullptr) {
            z += b_hh[h];
        }
        float out_val = (activation_type == 0) ? tanhf(z) : fmaxf(0.0f, z);
        h_out[idx] = out_val;
    }
}

// -----------------------------------------------------------------------------
// 3. Fused RNN Step Backward Kernel: dz_t = dh_total * d_act(h_t)
// -----------------------------------------------------------------------------
__global__ void rnn_step_backward_kernel(const float* __restrict__ dh_total,
                                         const float* __restrict__ h_t,
                                         float* __restrict__ dz_out,
                                         int size,
                                         int activation_type) { // 0 = tanh, 1 = relu
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dh = dh_total[idx];
        float h = h_t[idx];
        float dz = 0.0f;
        if (activation_type == 0) {
            // d/dz tanh(z) = 1 - tanh(z)^2 = 1 - h^2
            dz = dh * (1.0f - h * h);
        } else {
            // d/dz relu(z) = (h > 0) ? 1 : 0
            dz = (h > 0.0f) ? dh : 0.0f;
        }
        dz_out[idx] = dz;
    }
}

} // namespace rnn
} // namespace cuda_ml

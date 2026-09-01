#include "../include/cell_candidate_gate.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Cell Candidate Gate CUDA Kernels
// g_t = tanh(Z_g)
// dZ_g = dg_t * (1 - g_t^2)
// -----------------------------------------------------------------------------

__global__ void candidate_gate_forward_kernel(
    const float* __restrict__ d_Z_g,
    float* __restrict__ d_g,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_g[idx];
        d_g[idx] = tanhf(z);
    }
}

__global__ void candidate_gate_backward_kernel(
    const float* __restrict__ d_dg,
    const float* __restrict__ d_g,
    float* __restrict__ d_dZ_g,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dg = d_dg[idx];
        float g_val = d_g[idx];
        // dZ_g = dg * (1 - g^2)
        d_dZ_g[idx] = dg * (1.0f - g_val * g_val);
    }
}

void launch_candidate_gate_forward(
    const float* d_Z_g,
    float* d_g,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    candidate_gate_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z_g, d_g, size);
    CUDA_KERNEL_CHECK();
}

void launch_candidate_gate_backward(
    const float* d_dg,
    const float* d_g,
    float* d_dZ_g,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    candidate_gate_backward_kernel<<<blocks, threads, 0, stream>>>(d_dg, d_g, d_dZ_g, size);
    CUDA_KERNEL_CHECK();
}

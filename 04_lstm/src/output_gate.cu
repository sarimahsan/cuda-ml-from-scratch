#include "../include/output_gate.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Output Gate CUDA Kernels
// o_t = sigmoid(Z_o) = 1 / (1 + exp(-Z_o))
// dZ_o = do_t * o_t * (1 - o_t)
// -----------------------------------------------------------------------------

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void output_gate_forward_kernel(
    const float* __restrict__ d_Z_o,
    float* __restrict__ d_o,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_o[idx];
        d_o[idx] = sigmoidf_device(z);
    }
}

__global__ void output_gate_backward_kernel(
    const float* __restrict__ d_do,
    const float* __restrict__ d_o,
    float* __restrict__ d_dZ_o,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float do_val = d_do[idx];
        float o_val = d_o[idx];
        // dZ_o = do * o * (1 - o)
        d_dZ_o[idx] = do_val * o_val * (1.0f - o_val);
    }
}

void launch_output_gate_forward(
    const float* d_Z_o,
    float* d_o,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    output_gate_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z_o, d_o, size);
    CUDA_KERNEL_CHECK();
}

void launch_output_gate_backward(
    const float* d_do,
    const float* d_o,
    float* d_dZ_o,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    output_gate_backward_kernel<<<blocks, threads, 0, stream>>>(d_do, d_o, d_dZ_o, size);
    CUDA_KERNEL_CHECK();
}

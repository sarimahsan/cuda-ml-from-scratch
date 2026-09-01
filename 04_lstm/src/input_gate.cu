#include "../include/input_gate.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Input Gate CUDA Kernels
// i_t = sigmoid(Z_i) = 1 / (1 + exp(-Z_i))
// dZ_i = di_t * i_t * (1 - i_t)
// -----------------------------------------------------------------------------

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void input_gate_forward_kernel(
    const float* __restrict__ d_Z_i,
    float* __restrict__ d_i,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_i[idx];
        d_i[idx] = sigmoidf_device(z);
    }
}

__global__ void input_gate_backward_kernel(
    const float* __restrict__ d_di,
    const float* __restrict__ d_i,
    float* __restrict__ d_dZ_i,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float di = d_di[idx];
        float i_val = d_i[idx];
        // dZ_i = di * i * (1 - i)
        d_dZ_i[idx] = di * i_val * (1.0f - i_val);
    }
}

void launch_input_gate_forward(
    const float* d_Z_i,
    float* d_i,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    input_gate_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z_i, d_i, size);
    CUDA_KERNEL_CHECK();
}

void launch_input_gate_backward(
    const float* d_di,
    const float* d_i,
    float* d_dZ_i,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    input_gate_backward_kernel<<<blocks, threads, 0, stream>>>(d_di, d_i, d_dZ_i, size);
    CUDA_KERNEL_CHECK();
}

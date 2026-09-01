#include "../include/forget_gate.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Forget Gate CUDA Kernels
// f_t = sigmoid(Z_f) = 1 / (1 + exp(-Z_f))
// dZ_f = df_t * f_t * (1 - f_t)
// -----------------------------------------------------------------------------

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void forget_gate_forward_kernel(
    const float* __restrict__ d_Z_f,
    float* __restrict__ d_f,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_f[idx];
        d_f[idx] = sigmoidf_device(z);
    }
}

__global__ void forget_gate_backward_kernel(
    const float* __restrict__ d_df,
    const float* __restrict__ d_f,
    float* __restrict__ d_dZ_f,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float df = d_df[idx];
        float f_val = d_f[idx];
        // dZ_f = df * f * (1 - f)
        d_dZ_f[idx] = df * f_val * (1.0f - f_val);
    }
}

void launch_forget_gate_forward(
    const float* d_Z_f,
    float* d_f,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    forget_gate_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z_f, d_f, size);
    CUDA_KERNEL_CHECK();
}

void launch_forget_gate_backward(
    const float* d_df,
    const float* d_f,
    float* d_dZ_f,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    forget_gate_backward_kernel<<<blocks, threads, 0, stream>>>(d_df, d_f, d_dZ_f, size);
    CUDA_KERNEL_CHECK();
}

#include "../include/activations.cuh"
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608f // sqrt(2/pi)
#define COEFF 0.044715f

// -------------------------------------------------------------------------
// 1. Forward Activation Kernels
// -------------------------------------------------------------------------
__global__ void relu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = Z[idx];
        A[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        A[idx] = 1.0f / (1.0f + __expf(-Z[idx]));
    }
}

__global__ void gelu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        A[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void leaky_relu_forward_kernel(const float* __restrict__ Z, float* __restrict__ A, int size, float alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = Z[idx];
        A[idx] = val > 0.0f ? val : alpha * val;
    }
}

// -------------------------------------------------------------------------
// 2. Backward Activation Kernels
// -------------------------------------------------------------------------
__global__ void relu_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : 0.0f;
    }
}

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + __expf(-Z[idx]));
        dZ[idx] = dA[idx] * sig * (1.0f - sig);
    }
}

__global__ void gelu_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = Z[idx];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
        float d_gelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
        dZ[idx] = dA[idx] * d_gelu;
    }
}

__global__ void leaky_relu_backward_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ Z,
    float* __restrict__ dZ,
    int size,
    float alpha
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        dZ[idx] = (Z[idx] > 0.0f) ? dA[idx] : alpha * dA[idx];
    }
}

// -------------------------------------------------------------------------
// Host Launch Dispatchers
// -------------------------------------------------------------------------
void launch_activation_forward(
    const float* d_Z,
    float* d_A,
    int size,
    ActivationType act_type,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    switch (act_type) {
        case ActivationType::RELU:
            relu_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z, d_A, size);
            break;
        case ActivationType::SIGMOID:
            sigmoid_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z, d_A, size);
            break;
        case ActivationType::GELU:
            gelu_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z, d_A, size);
            break;
        case ActivationType::LEAKY_RELU:
            leaky_relu_forward_kernel<<<blocks, threads, 0, stream>>>(d_Z, d_A, size, 0.01f);
            break;
    }
}

void launch_activation_backward(
    const float* d_dA,
    const float* d_Z,
    float* d_dZ,
    int size,
    ActivationType act_type,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    switch (act_type) {
        case ActivationType::RELU:
            relu_backward_kernel<<<blocks, threads, 0, stream>>>(d_dA, d_Z, d_dZ, size);
            break;
        case ActivationType::SIGMOID:
            sigmoid_backward_kernel<<<blocks, threads, 0, stream>>>(d_dA, d_Z, d_dZ, size);
            break;
        case ActivationType::GELU:
            gelu_backward_kernel<<<blocks, threads, 0, stream>>>(d_dA, d_Z, d_dZ, size);
            break;
        case ActivationType::LEAKY_RELU:
            leaky_relu_backward_kernel<<<blocks, threads, 0, stream>>>(d_dA, d_Z, d_dZ, size, 0.01f);
            break;
    }
}

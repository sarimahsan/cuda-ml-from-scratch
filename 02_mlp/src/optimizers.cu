#include "../include/optimizers.cuh"
#include <cmath>

// -------------------------------------------------------------------------
// 1. SGD with Momentum In-Place Update Kernel
// -------------------------------------------------------------------------
__global__ void sgd_momentum_kernel(
    float* __restrict__ param,
    float* __restrict__ velocity,
    const float* __restrict__ grad,
    float lr,
    float momentum,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float v = momentum * velocity[idx] + grad[idx];
        velocity[idx] = v;
        param[idx] -= lr * v;
    }
}

// -------------------------------------------------------------------------
// 2. Adam In-Place Update Kernel
// -------------------------------------------------------------------------
__global__ void adam_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float bias_correction2,
    int size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;

        m[idx] = m_val;
        v[idx] = v_val;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        param[idx] -= (lr * m_hat) / (sqrtf(v_hat) + eps);
    }
}

// -------------------------------------------------------------------------
// Host Launch Functions
// -------------------------------------------------------------------------
void launch_sgd_momentum(
    float* d_param,
    float* d_velocity,
    const float* d_grad,
    float lr,
    float momentum,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_momentum_kernel<<<blocks, threads, 0, stream>>>(
        d_param, d_velocity, d_grad, lr, momentum, size
    );
}

void launch_adam(
    float* d_param,
    float* d_m,
    float* d_v,
    const float* d_grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    adam_kernel<<<blocks, threads, 0, stream>>>(
        d_param, d_m, d_v, d_grad,
        lr, beta1, beta2, eps,
        bias_correction1, bias_correction2,
        size
    );
}

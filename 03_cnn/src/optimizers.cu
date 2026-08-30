#include "../include/optimizers.cuh"
#include <cmath>

// -------------------------------------------------------------------------
// SGD with Momentum Kernel
// -------------------------------------------------------------------------
__global__ void sgd_momentum_kernel(
    float* __restrict__ param,
    float* __restrict__ velocity,
    const float* __restrict__ grad,
    int size,
    float lr,
    float momentum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = momentum * velocity[idx] + grad[idx];
        velocity[idx] = v;
        param[idx] -= lr * v;
    }
}

// -------------------------------------------------------------------------
// Adam Optimizer Kernel
// -------------------------------------------------------------------------
__global__ void adam_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bias_correction1,
    float bias_correction2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        float m_t = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_t = beta2 * v[idx] + (1.0f - beta2) * g * g;

        m[idx] = m_t;
        v[idx] = v_t;

        float m_hat = m_t / bias_correction1;
        float v_hat = v_t / bias_correction2;

        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// -------------------------------------------------------------------------
// Host Launchers
// -------------------------------------------------------------------------
void sgd_momentum_update(
    float* d_param,
    float* d_velocity,
    const float* d_grad,
    int size,
    float lr,
    float momentum,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (size + block - 1) / block;
    sgd_momentum_kernel<<<grid, block, 0, stream>>>(d_param, d_velocity, d_grad, size, lr, momentum);
    CUDA_CHECK_LAST();
}

void adam_update(
    float* d_param,
    float* d_m,
    float* d_v,
    const float* d_grad,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    cudaStream_t stream
) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    int block = 256;
    int grid = (size + block - 1) / block;
    adam_kernel<<<grid, block, 0, stream>>>(
        d_param, d_m, d_v, d_grad, size, lr, beta1, beta2, eps, bias_correction1, bias_correction2
    );
    CUDA_CHECK_LAST();
}

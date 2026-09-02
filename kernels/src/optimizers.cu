#include "../include/optimizers.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. SGD with Momentum Kernel
__global__ void sgd_momentum_kernel(float* __restrict__ param,
                                    float* __restrict__ velocity,
                                    const float* __restrict__ grad,
                                    float lr, float momentum, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float v = momentum * velocity[idx] + grad[idx];
        velocity[idx] = v;
        param[idx] -= lr * v;
    }
}

// 2. Adam In-Place Kernel
__global__ void adam_kernel(float* __restrict__ param,
                            float* __restrict__ m,
                            float* __restrict__ v,
                            const float* __restrict__ grad,
                            float lr, float beta1, float beta2, float eps,
                            float bias_correction1, float bias_correction2,
                            float weight_decay, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float p = param[idx];
        float g = grad[idx];

        if (weight_decay != 0.0f) {
            g += weight_decay * p;
        }

        float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;

        m[idx] = m_val;
        v[idx] = v_val;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        param[idx] = p - (lr * m_hat) / (sqrtf(v_hat) + eps);
    }
}

// 3. Gradient Norm Clipping Kernel
__global__ void clip_grad_norm_kernel(float* __restrict__ grad,
                                      float scale, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= scale;
    }
}

// Host Launchers
void sgd_momentum_step(float* param, float* velocity, const float* grad,
                       float lr, float momentum, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_momentum_kernel<<<blocks, threads, 0, stream>>>(
        param, velocity, grad, lr, momentum, size);
}

void adam_step(float* param, float* m, float* v, const float* grad,
               float lr, float beta1, float beta2, float eps,
               int step, int size, float weight_decay, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    adam_kernel<<<blocks, threads, 0, stream>>>(
        param, m, v, grad, lr, beta1, beta2, eps,
        bias_correction1, bias_correction2, weight_decay, size);
}

void clip_grad_norm(float* grad, int size, float max_norm, float actual_norm, cudaStream_t stream) {
    if (actual_norm > max_norm && actual_norm > 0.0f) {
        float scale = max_norm / (actual_norm + 1e-6f);
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        clip_grad_norm_kernel<<<blocks, threads, 0, stream>>>(grad, scale, size);
    }
}

} // namespace kernels
} // namespace cuda_ml

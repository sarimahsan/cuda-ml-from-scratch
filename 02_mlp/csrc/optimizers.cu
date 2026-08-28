#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
// PyTorch Extension Interface Functions
// -------------------------------------------------------------------------
void sgd_momentum_step_cuda(
    torch::Tensor param,
    torch::Tensor velocity,
    torch::Tensor grad,
    float lr,
    float momentum
) {
    int size = param.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sgd_momentum_kernel<<<blocks, threads>>>(
        param.data_ptr<float>(),
        velocity.data_ptr<float>(),
        grad.data_ptr<float>(),
        lr,
        momentum,
        size
    );
}

void adam_step_cuda(
    torch::Tensor param,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step
) {
    int size = param.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    adam_kernel<<<blocks, threads>>>(
        param.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        grad.data_ptr<float>(),
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        bias_correction2,
        size
    );
}

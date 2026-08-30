#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

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

void sgd_momentum_step_cuda(
    torch::Tensor param,
    torch::Tensor velocity,
    torch::Tensor grad,
    float lr,
    float momentum
) {
    TORCH_CHECK(param.is_cuda() && velocity.is_cuda() && grad.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(param.is_contiguous() && velocity.is_contiguous() && grad.is_contiguous(), "Inputs must be contiguous");

    int size = param.numel();
    int block = 256;
    int grid = (size + block - 1) / block;

    sgd_momentum_kernel<<<grid, block>>>(
        param.data_ptr<float>(),
        velocity.data_ptr<float>(),
        grad.data_ptr<float>(),
        size,
        lr,
        momentum
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
    TORCH_CHECK(param.is_cuda() && m.is_cuda() && v.is_cuda() && grad.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(param.is_contiguous() && m.is_contiguous() && v.is_contiguous() && grad.is_contiguous(), "Inputs must be contiguous");

    int size = param.numel();
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    int block = 256;
    int grid = (size + block - 1) / block;

    adam_kernel<<<grid, block>>>(
        param.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        grad.data_ptr<float>(),
        size,
        lr,
        beta1,
        beta2,
        eps,
        bias_correction1,
        bias_correction2
    );
}

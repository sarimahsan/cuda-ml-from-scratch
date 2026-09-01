#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void adam_kernel_torch(
    float* __restrict__ d_weights,
    float* __restrict__ d_m,
    float* __restrict__ d_v,
    const float* __restrict__ d_grads,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = d_grads[idx];
        float m = beta1 * d_m[idx] + (1.0f - beta1) * g;
        float v = beta2 * d_v[idx] + (1.0f - beta2) * g * g;

        d_m[idx] = m;
        d_v[idx] = v;

        float m_hat = m / bc1;
        float v_hat = v / bc2;

        d_weights[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

__global__ void sgd_momentum_kernel_torch(
    float* __restrict__ d_weights,
    float* __restrict__ d_v,
    const float* __restrict__ d_grads,
    int size,
    float lr,
    float momentum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = d_grads[idx];
        float v = momentum * d_v[idx] + g;
        d_v[idx] = v;
        d_weights[idx] -= lr * v;
    }
}

__global__ void scale_tensor_kernel_torch(
    float* __restrict__ d_tensor,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_tensor[idx] *= scale;
    }
}

void adam_step(
    torch::Tensor weight,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor grad,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step
) {
    TORCH_CHECK(weight.is_cuda() && grad.is_cuda(), "Tensors must be on CUDA");
    int size = weight.numel();
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    adam_kernel_torch<<<blocks, threads>>>(
        weight.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        grad.data_ptr<float>(),
        size, lr, beta1, beta2, eps, bc1, bc2
    );
}

void sgd_momentum_step(
    torch::Tensor weight,
    torch::Tensor v,
    torch::Tensor grad,
    float lr,
    float momentum
) {
    TORCH_CHECK(weight.is_cuda() && grad.is_cuda(), "Tensors must be on CUDA");
    int size = weight.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sgd_momentum_kernel_torch<<<blocks, threads>>>(
        weight.data_ptr<float>(),
        v.data_ptr<float>(),
        grad.data_ptr<float>(),
        size, lr, momentum
    );
}

void clip_grad_norm(
    std::vector<torch::Tensor> grads,
    float max_norm
) {
    if (max_norm <= 0.0f || grads.empty()) return;

    float total_sq_sum = 0.0f;
    for (const auto& g : grads) {
        if (g.defined() && g.numel() > 0) {
            total_sq_sum += g.norm(2).pow(2).item<float>();
        }
    }

    float total_norm = sqrtf(total_sq_sum);
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (auto& g : grads) {
            if (g.defined() && g.numel() > 0) {
                int size = g.numel();
                int threads = 256;
                int blocks = (size + threads - 1) / threads;
                scale_tensor_kernel_torch<<<blocks, threads>>>(
                    g.data_ptr<float>(), clip_coef, size
                );
            }
        }
    }
}

#include "../include/optimizers.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// Vectorized GPU Optimizers & Gradient Clipping
// -----------------------------------------------------------------------------

__global__ void sgd_momentum_kernel(
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

__global__ void adam_kernel(
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

__global__ void scale_tensor_kernel(
    float* __restrict__ d_tensor,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_tensor[idx] *= scale;
    }
}

__global__ void sum_sq_kernel(
    const float* __restrict__ d_tensor,
    float* __restrict__ d_sq_sum,
    int size
) {
    __shared__ float s_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float g = d_tensor[i];
        val += g * g;
    }
    s_sum[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_sq_sum, s_sum[0]);
    }
}

void launch_sgd_momentum_step(
    float* d_weights,
    float* d_v,
    const float* d_grads,
    int size,
    float lr,
    float momentum,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_momentum_kernel<<<blocks, threads, 0, stream>>>(d_weights, d_v, d_grads, size, lr, momentum);
    CUDA_KERNEL_CHECK();
}

void launch_adam_step(
    float* d_weights,
    float* d_m,
    float* d_v,
    const float* d_grads,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    cudaStream_t stream
) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    adam_kernel<<<blocks, threads, 0, stream>>>(
        d_weights, d_m, d_v, d_grads, size, lr, beta1, beta2, eps, bc1, bc2
    );
    CUDA_KERNEL_CHECK();
}

void launch_clip_grad_norm(
    float** d_grad_ptrs,
    const int* grad_sizes,
    int num_tensors,
    float max_norm,
    cudaStream_t stream
) {
    if (max_norm <= 0.0f) return;

    float* d_sq_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sq_sum, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_sq_sum, 0, sizeof(float), stream));

    for (int i = 0; i < num_tensors; ++i) {
        int size = grad_sizes[i];
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        if (blocks > 128) blocks = 128;
        sum_sq_kernel<<<blocks, threads, 0, stream>>>(d_grad_ptrs[i], d_sq_sum, size);
    }

    float h_sq_sum = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_sq_sum, d_sq_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_sq_sum));

    float total_norm = sqrtf(h_sq_sum);
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < num_tensors; ++i) {
            int size = grad_sizes[i];
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            scale_tensor_kernel<<<blocks, threads, 0, stream>>>(d_grad_ptrs[i], clip_coef, size);
        }
        CUDA_KERNEL_CHECK();
    }
}

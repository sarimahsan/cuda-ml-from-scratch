#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cfloat>

#define WARP_SIZE 32

__inline__ __device__ float softmax_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float softmax_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_cross_entropy_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ probs,
    float* __restrict__ dZ,
    float* __restrict__ sample_losses,
    int batch_size,
    int num_classes
) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_logits = logits + row * num_classes;
    const float* row_targets = targets + row * num_classes;
    float* row_probs = probs + row * num_classes;
    float* row_dZ = dZ + row * num_classes;

    // 1. Row Maximum
    float thread_max = -FLT_MAX;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[c]);
    }
    float row_max = softmax_warp_reduce_max(thread_max);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // 2. Exponentials sum
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float exp_val = expf(row_logits[c] - row_max);
        row_probs[c] = exp_val;
        thread_sum += exp_val;
    }
    float row_sum = softmax_warp_reduce_sum(thread_sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    row_sum = s_sum;

    // 3. Normalization, Cross-Entropy loss, and gradient computation
    float inv_sum = 1.0f / (row_sum + 1e-7f);
    float thread_loss = 0.0f;
    float inv_N = 1.0f / (float)batch_size;

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float p = row_probs[c] * inv_sum;
        row_probs[c] = p;
        float y = row_targets[c];
        row_dZ[c] = (p - y) * inv_N;
        if (y > 0.0f) {
            thread_loss += -y * logf(fmaxf(p, 1e-7f));
        }
    }

    float sample_loss = softmax_warp_reduce_sum(thread_loss);
    if (threadIdx.x == 0 && sample_losses != nullptr) {
        sample_losses[row] = sample_loss;
    }
}

__global__ void reduce_mean_loss_kernel(const float* sample_losses, float* total_loss, int batch_size) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        sum += sample_losses[i];
    }
    sum = softmax_warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *total_loss = sum / (float)batch_size;
    }
}

std::vector<torch::Tensor> softmax_cross_entropy_cuda(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous(), "Inputs must be contiguous");

    int batch_size = logits.size(0);
    int num_classes = logits.size(1);

    auto probs = torch::empty_like(logits);
    auto dZ = torch::empty_like(logits);
    auto sample_losses = torch::empty({batch_size}, logits.options());
    auto loss = torch::empty({1}, logits.options());

    int block = (num_classes <= 32) ? 32 : 128;

    softmax_cross_entropy_kernel<<<batch_size, block>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<float>(),
        probs.data_ptr<float>(),
        dZ.data_ptr<float>(),
        sample_losses.data_ptr<float>(),
        batch_size,
        num_classes
    );

    reduce_mean_loss_kernel<<<1, 32>>>(
        sample_losses.data_ptr<float>(),
        loss.data_ptr<float>(),
        batch_size
    );

    return {loss, probs, dZ};
}

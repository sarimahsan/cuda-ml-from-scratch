#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define WARP_SIZE 32
#define EPSILON 1e-7f

// -------------------------------------------------------------------------
// Reduction Helpers
// -------------------------------------------------------------------------
__inline__ __device__ float softmax_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float softmax_block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = softmax_warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        sum = softmax_warp_reduce_sum(sum);
    }
    return sum;
}

// -------------------------------------------------------------------------
// 1. Numerically Stable Softmax Forward Kernel
// -------------------------------------------------------------------------
__global__ void softmax_forward_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int N, int C
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        float max_val = -1e30f;
        for (int c = 0; c < C; ++c) {
            float v = logits[row * C + c];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float e = __expf(logits[row * C + c] - max_val);
            probs[row * C + c] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / (sum_exp + EPSILON);
        for (int c = 0; c < C; ++c) {
            probs[row * C + c] *= inv_sum;
        }
    }
}

// -------------------------------------------------------------------------
// 2. Categorical Cross-Entropy Loss & Analytical Gradient Kernel
// -------------------------------------------------------------------------
__global__ void cross_entropy_loss_and_grad_kernel(
    const float* __restrict__ probs,
    const int64_t* __restrict__ targets,
    float* __restrict__ dZ,
    float* __restrict__ block_losses,
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;

    if (idx < N) {
        int64_t target_label = targets[idx];
        if (target_label >= 0 && target_label < C) {
            float p = probs[idx * C + target_label];
            p = fmaxf(p, EPSILON);
            local_loss = -__logf(p);
        }

        // dZ = (probs - y_onehot) / N
        for (int c = 0; c < C; ++c) {
            float y_val = (c == target_label) ? 1.0f : 0.0f;
            dZ[idx * C + c] = (probs[idx * C + c] - y_val) / static_cast<float>(N);
        }
    }

    float block_sum = softmax_block_reduce_sum(local_loss);
    if (threadIdx.x == 0) {
        block_losses[blockIdx.x] = block_sum;
    }
}

__global__ void final_loss_reduction_kernel(
    const float* __restrict__ block_losses,
    float* __restrict__ total_loss,
    int num_blocks,
    int N
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_losses[i];
    }
    sum = softmax_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *total_loss = sum / static_cast<float>(N);
    }
}

// -------------------------------------------------------------------------
// PyTorch Extension Interface Functions
// -------------------------------------------------------------------------
std::vector<torch::Tensor> softmax_cross_entropy_cuda(torch::Tensor logits, torch::Tensor targets) {
    TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous(), "Inputs must be contiguous");

    int N = logits.size(0);
    int C = logits.size(1);

    auto probs = torch::empty({N, C}, logits.options());
    auto dZ    = torch::empty({N, C}, logits.options());

    int threads = 256;
    int blocks_sm = (N + threads - 1) / threads;

    // 1. Softmax probabilities
    softmax_forward_kernel<<<blocks_sm, threads>>>(
        logits.data_ptr<float>(),
        probs.data_ptr<float>(),
        N, C
    );

    // 2. Cross-Entropy Loss & Analytical Gradient dZ
    auto block_losses = torch::empty({blocks_sm}, logits.options());
    auto total_loss   = torch::empty({1}, logits.options());

    cross_entropy_loss_and_grad_kernel<<<blocks_sm, threads>>>(
        probs.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        dZ.data_ptr<float>(),
        block_losses.data_ptr<float>(),
        N, C
    );

    final_loss_reduction_kernel<<<1, 256>>>(
        block_losses.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        blocks_sm, N
    );

    return {probs, total_loss, dZ};
}

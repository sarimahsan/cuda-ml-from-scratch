#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max_t(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_t(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_max_t(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max_t(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max_t(val);
    return val;
}

__device__ float block_reduce_sum_t(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum_t(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_t(val);
    return val;
}

__global__ void sequence_softmax_cross_entropy_kernel_torch(
    const float* __restrict__ d_logits,
    const int64_t* __restrict__ d_targets,
    float* __restrict__ d_probs,
    float* __restrict__ d_losses,
    float* __restrict__ d_dlogits,
    int total_tokens,
    int vocab_size
) {
    int row = blockIdx.x;
    if (row >= total_tokens) return;

    __shared__ float s_max;
    __shared__ float s_sum;

    const float* row_logits = d_logits + row * vocab_size;
    float* row_probs = d_probs + row * vocab_size;
    float* row_dlogits = d_dlogits ? (d_dlogits + row * vocab_size) : nullptr;
    int64_t target = d_targets[row];

    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[col]);
    }
    float row_max = block_reduce_max_t(thread_max);
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        float exp_val = __expf(row_logits[col] - s_max);
        row_probs[col] = exp_val;
        thread_sum += exp_val;
    }
    float row_sum = block_reduce_sum_t(thread_sum);
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();

    float inv_sum = 1.0f / (s_sum + 1e-12f);
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        float prob = row_probs[col] * inv_sum;
        row_probs[col] = prob;
        if (row_dlogits) {
            float grad = (col == target) ? (prob - 1.0f) : prob;
            row_dlogits[col] = grad / (float)total_tokens;
        }
    }

    if (threadIdx.x == 0) {
        if (target >= 0 && target < vocab_size) {
            float target_p = row_probs[target];
            d_losses[row] = -__logf(fmaxf(target_p, 1e-12f));
        } else {
            d_losses[row] = 0.0f;
        }
    }
}

std::vector<torch::Tensor> softmax_cross_entropy(
    torch::Tensor logits,
    torch::Tensor targets
) {
    TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "Inputs must be CUDA tensors");
    int total_tokens = logits.size(0);
    int vocab_size = logits.size(1);

    auto probs = torch::empty_like(logits);
    auto losses = torch::empty({total_tokens}, logits.options());
    auto dlogits = torch::empty_like(logits);

    int threads = (vocab_size < 256) ? 128 : 256;
    if (threads > 1024) threads = 1024;

    sequence_softmax_cross_entropy_kernel_torch<<<total_tokens, threads>>>(
        logits.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        probs.data_ptr<float>(),
        losses.data_ptr<float>(),
        dlogits.data_ptr<float>(),
        total_tokens,
        vocab_size
    );

    auto mean_loss = losses.mean();
    return {probs, mean_loss, dlogits};
}

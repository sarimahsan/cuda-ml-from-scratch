#include "../include/softmax_loss.cuh"
#include <cfloat>
#include <cmath>

// -----------------------------------------------------------------------------
// Sequence Softmax Cross-Entropy Loss with Warp-Level Reductions
// Each block processes 1 or more token rows.
// -----------------------------------------------------------------------------

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level maximum reduction
__device__ float block_reduce_max(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

// Block-level summation reduction
__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void sequence_softmax_cross_entropy_kernel(
    const float* __restrict__ d_logits,
    const int* __restrict__ d_targets,
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
    int target = d_targets[row];

    // 1. Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        thread_max = fmaxf(thread_max, row_logits[col]);
    }
    float row_max = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();

    // 2. Compute exp and sum
    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        float exp_val = __expf(row_logits[col] - s_max);
        row_probs[col] = exp_val;
        thread_sum += exp_val;
    }
    float row_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();

    // 3. Normalize to probabilities and compute analytical gradient
    float inv_sum = 1.0f / (s_sum + 1e-12f);
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        float prob = row_probs[col] * inv_sum;
        row_probs[col] = prob;
        if (row_dlogits) {
            float grad = (col == target) ? (prob - 1.0f) : prob;
            // Scale by 1 / total_tokens for mean batch loss gradient
            row_dlogits[col] = grad / (float)total_tokens;
        }
    }

    // 4. Compute cross-entropy loss for target token: -log(P(target))
    if (threadIdx.x == 0) {
        if (target >= 0 && target < vocab_size) {
            float target_p = row_probs[target];
            d_losses[row] = -__logf(fmaxf(target_p, 1e-12f));
        } else {
            d_losses[row] = 0.0f;
        }
    }
}

__global__ void reduce_mean_loss_kernel(
    const float* __restrict__ d_losses,
    float* __restrict__ d_mean_loss,
    int total_tokens
) {
    float thread_sum = 0.0f;
    for (int idx = threadIdx.x; idx < total_tokens; idx += blockDim.x) {
        thread_sum += d_losses[idx];
    }
    float total_loss = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) {
        *d_mean_loss = total_loss / (float)total_tokens;
    }
}

void launch_sequence_softmax_cross_entropy(
    const float* d_logits,
    const int* d_targets,
    float* d_probs,
    float* d_losses,
    float* d_dlogits,
    int total_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    int threads = (vocab_size < 256) ? 128 : 256;
    if (threads > 1024) threads = 1024;
    sequence_softmax_cross_entropy_kernel<<<total_tokens, threads, 0, stream>>>(
        d_logits, d_targets, d_probs, d_losses, d_dlogits, total_tokens, vocab_size
    );
    CUDA_KERNEL_CHECK();
}

float compute_mean_loss(
    const float* d_losses,
    int total_tokens,
    cudaStream_t stream
) {
    float* d_mean = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mean, sizeof(float)));
    reduce_mean_loss_kernel<<<1, 256, 0, stream>>>(d_losses, d_mean, total_tokens);
    CUDA_KERNEL_CHECK();

    float h_mean = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mean));
    return h_mean;
}

#include "../include/softmax.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. Numerically Stable Warp-Level Softmax Forward Pass (1 warp per row) with 128-bit Vectorization
__global__ void softmax_forward_kernel(const float* __restrict__ logits,
                                       float* __restrict__ probs,
                                       int N, int C) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_logits = logits + row * C;
        float* row_probs = probs + row * C;

        int c4 = C / 4;
        const float4* row_logits4 = reinterpret_cast<const float4*>(row_logits);
        float4* row_probs4 = reinterpret_cast<float4*>(row_probs);

        // Step 1: Find row maximum using float4 vectorized loads
        float max_val = -1e20f;
        for (int i = lane; i < c4; i += 32) {
            float4 v = __ldg(&row_logits4[i]);
            max_val = fmaxf(max_val, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        for (int c = c4 * 4 + lane; c < C; c += 32) {
            max_val = fmaxf(max_val, row_logits[c]);
        }
        max_val = warp_reduce_max(max_val);
        max_val = __shfl_sync(FULL_MASK, max_val, 0);

        // Step 2: Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = lane; i < c4; i += 32) {
            float4 v = __ldg(&row_logits4[i]);
            sum_exp += expf(v.x - max_val) + expf(v.y - max_val) + expf(v.z - max_val) + expf(v.w - max_val);
        }
        for (int c = c4 * 4 + lane; c < C; c += 32) {
            sum_exp += expf(row_logits[c] - max_val);
        }
        sum_exp = warp_reduce_sum(sum_exp);
        sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);

        // Step 3: Normalize and write out using float4 vectorized stores
        float inv_sum = __fdividef(1.0f, sum_exp + 1e-12f);
        for (int i = lane; i < c4; i += 32) {
            float4 v = __ldg(&row_logits4[i]);
            float4 out;
            out.x = expf(v.x - max_val) * inv_sum;
            out.y = expf(v.y - max_val) * inv_sum;
            out.z = expf(v.z - max_val) * inv_sum;
            out.w = expf(v.w - max_val) * inv_sum;
            row_probs4[i] = out;
        }
        for (int c = c4 * 4 + lane; c < C; c += 32) {
            row_probs[c] = expf(row_logits[c] - max_val) * inv_sum;
        }
    }
}

// 2. Online Safe FlashSoftmax with 128-bit Vectorization
__global__ void online_safe_softmax_kernel(const float* __restrict__ logits,
                                           float* __restrict__ probs,
                                           int N, int C) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_logits = logits + row * C;
        float* row_probs = probs + row * C;

        int c4 = C / 4;
        const float4* row_logits4 = reinterpret_cast<const float4*>(row_logits);
        float4* row_probs4 = reinterpret_cast<float4*>(row_probs);

        float m_i = -1e20f;
        float d_i = 0.0f;

        // Online safe max & exp-sum tracking with float4 loads
        for (int i = lane; i < c4; i += 32) {
            float4 v = __ldg(&row_logits4[i]);
            
            float m_prev = m_i;
            m_i = fmaxf(m_i, v.x);
            d_i = d_i * expf(m_prev - m_i) + expf(v.x - m_i);

            m_prev = m_i;
            m_i = fmaxf(m_i, v.y);
            d_i = d_i * expf(m_prev - m_i) + expf(v.y - m_i);

            m_prev = m_i;
            m_i = fmaxf(m_i, v.z);
            d_i = d_i * expf(m_prev - m_i) + expf(v.z - m_i);

            m_prev = m_i;
            m_i = fmaxf(m_i, v.w);
            d_i = d_i * expf(m_prev - m_i) + expf(v.w - m_i);
        }

        for (int c = c4 * 4 + lane; c < C; c += 32) {
            float x = row_logits[c];
            float m_prev = m_i;
            m_i = fmaxf(m_i, x);
            d_i = d_i * expf(m_prev - m_i) + expf(x - m_i);
        }

        // Warp-level online reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float m_other = __shfl_down_sync(FULL_MASK, m_i, offset);
            float d_other = __shfl_down_sync(FULL_MASK, d_i, offset);
            float m_max = fmaxf(m_i, m_other);
            d_i = d_i * expf(m_i - m_max) + d_other * expf(m_other - m_max);
            m_i = m_max;
        }

        m_i = __shfl_sync(FULL_MASK, m_i, 0);
        d_i = __shfl_sync(FULL_MASK, d_i, 0);

        float inv_d = __fdividef(1.0f, d_i + 1e-12f);

        for (int i = lane; i < c4; i += 32) {
            float4 v = __ldg(&row_logits4[i]);
            float4 out;
            out.x = expf(v.x - m_i) * inv_d;
            out.y = expf(v.y - m_i) * inv_d;
            out.z = expf(v.z - m_i) * inv_d;
            out.w = expf(v.w - m_i) * inv_d;
            row_probs4[i] = out;
        }

        for (int c = c4 * 4 + lane; c < C; c += 32) {
            row_probs[c] = expf(row_logits[c] - m_i) * inv_d;
        }
    }
}

// 3. LogSoftmax Forward Pass
__global__ void log_softmax_forward_kernel(const float* __restrict__ logits,
                                           float* __restrict__ log_probs,
                                           int N, int C) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_logits = logits + row * C;
        float* row_out = log_probs + row * C;

        float max_val = -1e20f;
        for (int c = lane; c < C; c += 32) {
            max_val = fmaxf(max_val, row_logits[c]);
        }
        max_val = warp_reduce_max(max_val);
        max_val = __shfl_sync(FULL_MASK, max_val, 0);

        float sum_exp = 0.0f;
        for (int c = lane; c < C; c += 32) {
            sum_exp += expf(row_logits[c] - max_val);
        }
        sum_exp = warp_reduce_sum(sum_exp);
        sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);

        float log_sum = logf(sum_exp + 1e-12f);
        for (int c = lane; c < C; c += 32) {
            row_out[c] = (row_logits[c] - max_val) - log_sum;
        }
    }
}

// 4. Softmax Backward Pass: dLogits = Probs * (dProbs - sum(dProbs * Probs))
__global__ void softmax_backward_kernel(const float* __restrict__ dProbs,
                                        const float* __restrict__ probs,
                                        float* __restrict__ dLogits,
                                        int N, int C) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_dProbs = dProbs + row * C;
        const float* row_probs = probs + row * C;
        float* row_dLogits = dLogits + row * C;

        float sum_dp = 0.0f;
        for (int c = lane; c < C; c += 32) {
            sum_dp += row_dProbs[c] * row_probs[c];
        }
        sum_dp = warp_reduce_sum(sum_dp);
        sum_dp = __shfl_sync(FULL_MASK, sum_dp, 0);

        for (int c = lane; c < C; c += 32) {
            row_dLogits[c] = row_probs[c] * (row_dProbs[c] - sum_dp);
        }
    }
}

// 5. Fused Softmax + Cross-Entropy Forward & Backward
__global__ void fused_softmax_ce_kernel(const float* __restrict__ logits,
                                        const int64_t* __restrict__ targets,
                                        float* __restrict__ loss,
                                        float* __restrict__ dLogits,
                                        int N, int C) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_logits = logits + row * C;
        float* row_dLogits = dLogits + row * C;
        int64_t target = targets[row];

        float max_val = -1e20f;
        for (int c = lane; c < C; c += 32) {
            max_val = fmaxf(max_val, row_logits[c]);
        }
        max_val = warp_reduce_max(max_val);
        max_val = __shfl_sync(FULL_MASK, max_val, 0);

        float sum_exp = 0.0f;
        for (int c = lane; c < C; c += 32) {
            sum_exp += expf(row_logits[c] - max_val);
        }
        sum_exp = warp_reduce_sum(sum_exp);
        sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);

        float inv_sum = 1.0f / (sum_exp + 1e-12f);

        // Gradient: dLogits = (p_i - 1_{i=target}) / N
        for (int c = lane; c < C; c += 32) {
            float p = expf(row_logits[c] - max_val) * inv_sum;
            float indicator = (c == target) ? 1.0f : 0.0f;
            row_dLogits[c] = (p - indicator) / (float)N;
        }

        // Loss: -log(p_target)
        if (lane == 0) {
            float p_target = expf(row_logits[target] - max_val) * inv_sum;
            float row_loss = -logf(fmaxf(p_target, 1e-12f)) / (float)N;
            atomicAdd(loss, row_loss);
        }
    }
}

// Host Launchers
void softmax_forward(const float* logits, float* probs, int N, int C, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    softmax_forward_kernel<<<blocks, threads, 0, stream>>>(logits, probs, N, C);
}

void online_safe_softmax_forward(const float* logits, float* probs, int N, int C, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    online_safe_softmax_kernel<<<blocks, threads, 0, stream>>>(logits, probs, N, C);
}

void log_softmax_forward(const float* logits, float* log_probs, int N, int C, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    log_softmax_forward_kernel<<<blocks, threads, 0, stream>>>(logits, log_probs, N, C);
}

void softmax_backward(const float* dProbs, const float* probs, float* dLogits,
                      int N, int C, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    softmax_backward_kernel<<<blocks, threads, 0, stream>>>(dProbs, probs, dLogits, N, C);
}

void fused_softmax_cross_entropy_forward_backward(const float* logits, const int64_t* targets,
                                                  float* loss, float* dLogits,
                                                  int N, int C, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), stream));
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    fused_softmax_ce_kernel<<<blocks, threads, 0, stream>>>(logits, targets, loss, dLogits, N, C);
}

} // namespace kernels
} // namespace cuda_ml

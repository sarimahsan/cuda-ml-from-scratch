#include "../include/normalization.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. LayerNorm Forward with Welford Algorithm, 128-bit Vectorization, and Single-Pass Register Caching
__global__ void layernorm_forward_kernel(const float* __restrict__ X,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ Y,
                                         float* __restrict__ save_mean,
                                         float* __restrict__ save_rstd,
                                         int N, int D, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_x = X + row * D;
        float* row_y = Y + row * D;

        int d4 = ((D & 3) == 0) ? (D / 4) : 0;
        const float4* row_x4 = (d4 > 0) ? reinterpret_cast<const float4*>(row_x) : nullptr;
        float4* row_y4 = (d4 > 0) ? reinterpret_cast<float4*>(row_y) : nullptr;
        const float4* gamma4 = (gamma && d4 > 0) ? reinterpret_cast<const float4*>(gamma) : nullptr;
        const float4* beta4  = (beta && d4 > 0)  ? reinterpret_cast<const float4*>(beta)  : nullptr;

        // Register cache for up to D = 1024 (8 float4s per thread)
        constexpr int MAX_VECS = 8;
        float4 reg_cache[MAX_VECS];

        // Welford algorithm for online mean and variance
        float mean = 0.0f;
        float m2 = 0.0f;
        float count = 0.0f;

        int vec_idx = 0;
        for (int i = lane; i < d4; i += 32) {
            float4 v = __ldg(&row_x4[i]);
            if (vec_idx < MAX_VECS) {
                reg_cache[vec_idx++] = v;
            }

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float x = (k == 0) ? v.x : ((k == 1) ? v.y : ((k == 2) ? v.z : v.w));
                count += 1.0f;
                float delta = x - mean;
                mean += delta / count;
                m2 += delta * (x - mean);
            }
        }

        for (int d = d4 * 4 + lane; d < D; d += 32) {
            float x = row_x[d];
            count += 1.0f;
            float delta = x - mean;
            mean += delta / count;
            m2 += delta * (x - mean);
        }

        // Warp-level Welford combine
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float count_b = __shfl_down_sync(FULL_MASK, count, offset);
            float mean_b  = __shfl_down_sync(FULL_MASK, mean, offset);
            float m2_b    = __shfl_down_sync(FULL_MASK, m2, offset);

            float total_count = count + count_b;
            if (total_count > 0.0f) {
                float delta = mean_b - mean;
                mean += delta * (count_b / total_count);
                m2 += m2_b + delta * delta * (count * count_b / total_count);
                count = total_count;
            }
        }

        mean = __shfl_sync(FULL_MASK, mean, 0);
        float var = __shfl_sync(FULL_MASK, m2 / (float)D, 0);
        float rstd = rsqrtf(var + eps);

        if (lane == 0) {
            if (save_mean) save_mean[row] = mean;
            if (save_rstd) save_rstd[row] = rstd;
        }

        // Normalize and scale directly from cached registers!
        vec_idx = 0;
        for (int i = lane; i < d4; i += 32) {
            float4 v = (vec_idx < MAX_VECS) ? reg_cache[vec_idx++] : __ldg(&row_x4[i]);
            float4 g = gamma4 ? __ldg(&gamma4[i]) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 b = beta4  ? __ldg(&beta4[i])  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            float4 out;
            out.x = (v.x - mean) * rstd * g.x + b.x;
            out.y = (v.y - mean) * rstd * g.y + b.y;
            out.z = (v.z - mean) * rstd * g.z + b.z;
            out.w = (v.w - mean) * rstd * g.w + b.w;
            row_y4[i] = out;
        }

        for (int d = d4 * 4 + lane; d < D; d += 32) {
            float x_hat = (row_x[d] - mean) * rstd;
            float g = (gamma != nullptr) ? gamma[d] : 1.0f;
            float b = (beta != nullptr) ? beta[d] : 0.0f;
            row_y[d] = x_hat * g + b;
        }
    }
}

// LayerNorm Backward Kernel
__global__ void layernorm_backward_kernel(const float* __restrict__ dY,
                                          const float* __restrict__ X,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ rstd,
                                          float* __restrict__ dX,
                                          float* __restrict__ dgamma,
                                          float* __restrict__ dbeta,
                                          int N, int D) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_dy = dY + row * D;
        const float* row_x  = X + row * D;
        float* row_dx       = dX + row * D;

        float mu = mean[row];
        float rs = rstd[row];

        float sum_dy_xhat = 0.0f;
        float sum_dy_gamma = 0.0f;

        for (int d = lane; d < D; d += 32) {
            float dy = row_dy[d];
            float x_hat = (row_x[d] - mu) * rs;
            float g = (gamma != nullptr) ? gamma[d] : 1.0f;

            sum_dy_xhat += dy * g * x_hat;
            sum_dy_gamma += dy * g;

            if (dgamma != nullptr) atomicAdd(&dgamma[d], dy * x_hat);
            if (dbeta != nullptr) atomicAdd(&dbeta[d], dy);
        }

        sum_dy_xhat  = warp_reduce_sum(sum_dy_xhat);
        sum_dy_gamma = warp_reduce_sum(sum_dy_gamma);
        sum_dy_xhat  = __shfl_sync(FULL_MASK, sum_dy_xhat, 0);
        sum_dy_gamma = __shfl_sync(FULL_MASK, sum_dy_gamma, 0);

        float inv_D = 1.0f / (float)D;

        for (int d = lane; d < D; d += 32) {
            float dy = row_dy[d];
            float x_hat = (row_x[d] - mu) * rs;
            float g = (gamma != nullptr) ? gamma[d] : 1.0f;

            row_dx[d] = rs * (dy * g - (sum_dy_gamma + x_hat * sum_dy_xhat) * inv_D);
        }
    }
}

// 2. RMSNorm Forward Kernel
__global__ void rmsnorm_forward_kernel(const float* __restrict__ X,
                                       const float* __restrict__ gamma,
                                       float* __restrict__ Y,
                                       float* __restrict__ save_rstd,
                                       int N, int D, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < N) {
        const float* row_x = X + row * D;
        float* row_y = Y + row * D;

        float sum_sq = 0.0f;
        for (int d = lane; d < D; d += 32) {
            float v = row_x[d];
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        sum_sq = __shfl_sync(FULL_MASK, sum_sq, 0);

        float rms = rsqrtf(sum_sq / (float)D + eps);
        if (lane == 0 && save_rstd) {
            save_rstd[row] = rms;
        }

        for (int d = lane; d < D; d += 32) {
            float g = (gamma != nullptr) ? gamma[d] : 1.0f;
            row_y[d] = row_x[d] * rms * g;
        }
    }
}

// Host Launchers
void layernorm_forward(const float* X, const float* gamma, const float* beta,
                       float* Y, float* mean, float* rstd,
                       int N, int D, float eps, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    layernorm_forward_kernel<<<blocks, threads, 0, stream>>>(
        X, gamma, beta, Y, mean, rstd, N, D, eps);
}

void layernorm_backward(const float* dY, const float* X, const float* gamma,
                        const float* mean, const float* rstd,
                        float* dX, float* dgamma, float* dbeta,
                        int N, int D, cudaStream_t stream) {
    if (dgamma) CUDA_CHECK(cudaMemsetAsync(dgamma, 0, D * sizeof(float), stream));
    if (dbeta) CUDA_CHECK(cudaMemsetAsync(dbeta, 0, D * sizeof(float), stream));

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    layernorm_backward_kernel<<<blocks, threads, 0, stream>>>(
        dY, X, gamma, mean, rstd, dX, dgamma, dbeta, N, D);
}

void rmsnorm_forward(const float* X, const float* gamma, float* Y, float* rstd,
                     int N, int D, float eps, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;
    rmsnorm_forward_kernel<<<blocks, threads, 0, stream>>>(
        X, gamma, Y, rstd, N, D, eps);
}

} // namespace kernels
} // namespace cuda_ml

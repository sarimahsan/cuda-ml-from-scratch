#include "../include/reduction.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. Single-pass / Multi-block Array Sum Reduction using warp shuffle & atomicAdd
__global__ void reduce_sum_kernel(const float* __restrict__ X, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += X[i];
    }

    sum = block_reduce_sum<256>(sum);

    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

// 2. Array Max Reduction
__global__ void reduce_max_kernel(const float* __restrict__ X, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float max_val = -1e20f;
    for (int i = idx; i < N; i += stride) {
        max_val = fmaxf(max_val, X[i]);
    }

    max_val = block_reduce_max<256>(max_val);

    if (threadIdx.x == 0) {
        // Atomic max for float using CAS
        int* address_as_int = (int*)out;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                __float_as_int(fmaxf(max_val, __int_as_float(assumed))));
        } while (assumed != old);
    }
}

// 3. Vector L2 Norm: out = sum(x^2)
__global__ void reduce_l2_sq_kernel(const float* __restrict__ X, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum_sq = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float v = X[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum<256>(sum_sq);

    if (threadIdx.x == 0) {
        atomicAdd(out, sum_sq);
    }
}

// 4. Row-wise reductions: X is (M x N), 1 warp per row
__global__ void reduce_row_sum_kernel(const float* __restrict__ X, float* __restrict__ out, int M, int N) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < M) {
        float sum = 0.0f;
        for (int col = lane; col < N; col += 32) {
            sum += X[row * N + col];
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

__global__ void reduce_row_max_kernel(const float* __restrict__ X, float* __restrict__ out, int M, int N) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;

    if (row < M) {
        float max_v = -1e20f;
        for (int col = lane; col < N; col += 32) {
            max_v = fmaxf(max_v, X[row * N + col]);
        }
        max_v = warp_reduce_max(max_v);
        if (lane == 0) {
            out[row] = max_v;
        }
    }
}

// 5. Column-wise reductions: X is (M x N), out is (N)
__global__ void reduce_col_sum_kernel(const float* __restrict__ X, float* __restrict__ out, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += X[row * N + col];
        }
        out[col] = sum;
    }
}

// Host Launchers
void reduce_sum(const float* X, float* out, int N, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    int blocks = std::min(1024, (N + 255) / 256);
    reduce_sum_kernel<<<blocks, 256, 0, stream>>>(X, out, N);
}

void reduce_max(const float* X, float* out, int N, cudaStream_t stream) {
    float init_val = -1e20f;
    CUDA_CHECK(cudaMemcpyAsync(out, &init_val, sizeof(float), cudaMemcpyHostToDevice, stream));
    int blocks = std::min(1024, (N + 255) / 256);
    reduce_max_kernel<<<blocks, 256, 0, stream>>>(X, out, N);
}

void reduce_l2_norm(const float* X, float* out, int N, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    int blocks = std::min(1024, (N + 255) / 256);
    reduce_l2_sq_kernel<<<blocks, 256, 0, stream>>>(X, out, N);
}

void reduce_row_sum(const float* X, float* out, int M, int N, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    reduce_row_sum_kernel<<<blocks, threads, 0, stream>>>(X, out, M, N);
}

void reduce_row_max(const float* X, float* out, int M, int N, cudaStream_t stream) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    reduce_row_max_kernel<<<blocks, threads, 0, stream>>>(X, out, M, N);
}

void reduce_col_sum(const float* X, float* out, int M, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_col_sum_kernel<<<blocks, threads, 0, stream>>>(X, out, M, N);
}

} // namespace kernels
} // namespace cuda_ml

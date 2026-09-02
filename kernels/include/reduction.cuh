#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// High-Performance Reduction Engine (Warp-Shuffles + Multi-Block Aggregation)
// ============================================================================

// 1. Global Array Sum Reduction: out = sum(X)
void reduce_sum(const float* X, float* out, int N, cudaStream_t stream = 0);

// 2. Global Array Max / Min Reduction: out = max(X)
void reduce_max(const float* X, float* out, int N, cudaStream_t stream = 0);
void reduce_min(const float* X, float* out, int N, cudaStream_t stream = 0);

// 3. Vector L2 Norm: out = sqrt(sum(X_i^2))
void reduce_l2_norm(const float* X, float* out, int N, cudaStream_t stream = 0);

// 4. Row-wise Reductions: X is (M x N), out is (M)
void reduce_row_sum(const float* X, float* out, int M, int N, cudaStream_t stream = 0);
void reduce_row_max(const float* X, float* out, int M, int N, cudaStream_t stream = 0);
void reduce_row_mean(const float* X, float* out, int M, int N, cudaStream_t stream = 0);

// 5. Column-wise Reductions: X is (M x N), out is (N) (e.g. Bias gradients)
void reduce_col_sum(const float* X, float* out, int M, int N, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

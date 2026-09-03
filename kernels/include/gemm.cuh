#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// GEMM Kernels: C = alpha * A * B + beta * C
// A: (M x K), B: (K x N), C: (M x N)
// ============================================================================

// 1. Naive Baseline GEMM
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K,
                float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// 2. 2D Shared-Memory Tiled GEMM (Tile = 16x16 or 32x32 with zero bank conflict padding)
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// 3. Register-Tiled High-Throughput GEMM (Each thread computes 4x4 or 8x8 micro-tile)
void gemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// 4. Split-K GEMM for Reduction-Heavy Shapes (K >> M, N)
void gemm_split_k(const float* A, const float* B, float* C, int M, int N, int K,
                  int split_k = 8, float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// 5. Batched GEMM for Sequence / Multi-Head Operations: (B x M x K) * (B x K x N)
void gemm_batched(const float* A, const float* B, float* C, int batch_size,
                 int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                 cudaStream_t stream = 0);

// 5. Transposed GEMMs (commonly needed in Backpropagation)
// C = A * B^T  (A: M x K, B: N x K, C: M x N)
void gemm_NT(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

// C = A^T * B  (A: K x M, B: K x N, C: M x N)
void gemm_TN(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha = 1.0f, float beta = 0.0f, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

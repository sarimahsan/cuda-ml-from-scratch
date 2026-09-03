#include "../include/gemm.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// 1. Naive Baseline GEMM: C = alpha * A * B + beta * C
// ============================================================================
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        if (beta == 0.0f) {
            C[row * N + col] = alpha * sum;
        } else {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

// ============================================================================
// 2. 2D Shared Memory Tiled GEMM (TILE_DIM = 32)
// ============================================================================
#define TILE_DIM 32

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  float alpha, float beta) {
    __shared__ float s_A[TILE_DIM][TILE_DIM + 1]; // +1 padding prevents bank conflicts
    __shared__ float s_B[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;

        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}

// ============================================================================
// 3. High-Performance GEMM: Double Buffering + Warp Tiling + Vectorized Epilogue
// Tile: BM=128, BN=128, BK=8 | Warp: 64x32 (8 warps) | Thread: TM=8, TN=8
// ============================================================================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_double_buffered_warp_tiled_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int N, int K,
                                                      float alpha, float beta) {
    // Ping-pong shared memory buffers
    __shared__ float s_A[2][BM][BK + 1];
    __shared__ float s_B[2][BK][BN + 1];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    // Warp and Lane hierarchy (8 warps arranged as 2x4 grid of 64x32 warp tiles)
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    int warpRow = (warpId / 4) * 64;
    int warpCol = (warpId % 4) * 32;

    int laneRow = (laneId / 4) * TM;
    int laneCol = (laneId % 4) * TN;

    int threadRowInBlock = warpRow + laneRow;
    int threadColInBlock = warpCol + laneCol;

    // 64 register accumulators
    float r_c[TM][TN] = {0.0f};

    // Vectorized load helper indices (256 threads loading 1024 elements = float4 per thread)
    int a_load_row = threadIdx.x / 2;
    int a_load_col = (threadIdx.x % 2) * 4;

    int b_load_row = threadIdx.x / 32;
    int b_load_col = (threadIdx.x % 32) * 4;

    // Helper lambda for loading 1 float4 tile into shared memory buffer
    auto load_tile = [&](int buffer_idx, int bk) {
        // Load s_A
        if (cRow + a_load_row < M && bk + a_load_col + 3 < K) {
            float4 a_val = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (bk + a_load_col)]));
            s_A[buffer_idx][a_load_row][a_load_col + 0] = a_val.x;
            s_A[buffer_idx][a_load_row][a_load_col + 1] = a_val.y;
            s_A[buffer_idx][a_load_row][a_load_col + 2] = a_val.z;
            s_A[buffer_idx][a_load_row][a_load_col + 3] = a_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (cRow + a_load_row < M && bk + a_load_col + i < K) {
                    s_A[buffer_idx][a_load_row][a_load_col + i] = A[(cRow + a_load_row) * K + (bk + a_load_col + i)];
                } else {
                    s_A[buffer_idx][a_load_row][a_load_col + i] = 0.0f;
                }
            }
        }

        // Load s_B
        if (bk + b_load_row < K && cCol + b_load_col + 3 < N) {
            float4 b_val = __ldg(reinterpret_cast<const float4*>(&B[(bk + b_load_row) * N + (cCol + b_load_col)]));
            s_B[buffer_idx][b_load_row][b_load_col + 0] = b_val.x;
            s_B[buffer_idx][b_load_row][b_load_col + 1] = b_val.y;
            s_B[buffer_idx][b_load_row][b_load_col + 2] = b_val.z;
            s_B[buffer_idx][b_load_row][b_load_col + 3] = b_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (bk + b_load_row < K && cCol + b_load_col + i < N) {
                    s_B[buffer_idx][b_load_row][b_load_col + i] = B[(bk + b_load_row) * N + (cCol + b_load_col + i)];
                } else {
                    s_B[buffer_idx][b_load_row][b_load_col + i] = 0.0f;
                }
            }
        }
    };

    // --- PROLOGUE: Load Tile 0 into Buffer 0 ---
    load_tile(0, 0);
    __syncthreads();

    int num_tiles = (K + BK - 1) / BK;

    // --- MAIN PIPELINED LOOP (Double Buffering) ---
    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = (t + 1) & 1;
        int next_bk = (t + 1) * BK;

        float4 a_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 1. Asynchronously prefetch next tile into local thread registers
        if (t + 1 < num_tiles) {
            if (cRow + a_load_row < M && next_bk + a_load_col + 3 < K) {
                a_prefetch = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (next_bk + a_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (cRow + a_load_row < M && next_bk + a_load_col + i < K) {
                        float v = A[(cRow + a_load_row) * K + (next_bk + a_load_col + i)];
                        if (i == 0) a_prefetch.x = v;
                        else if (i == 1) a_prefetch.y = v;
                        else if (i == 2) a_prefetch.z = v;
                        else if (i == 3) a_prefetch.w = v;
                    }
                }
            }

            if (next_bk + b_load_row < K && cCol + b_load_col + 3 < N) {
                b_prefetch = __ldg(reinterpret_cast<const float4*>(&B[(next_bk + b_load_row) * N + (cCol + b_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (next_bk + b_load_row < K && cCol + b_load_col + i < N) {
                        float v = B[(next_bk + b_load_row) * N + (cCol + b_load_col + i)];
                        if (i == 0) b_prefetch.x = v;
                        else if (i == 1) b_prefetch.y = v;
                        else if (i == 2) b_prefetch.z = v;
                        else if (i == 3) b_prefetch.w = v;
                    }
                }
            }
        }

        // 2. Compute FMA outer products on current buffer while prefetch is in flight
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float r_a[TM];
            float r_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_a[i] = s_A[cur][threadRowInBlock + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_b[j] = s_B[cur][k][threadColInBlock + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        // 3. Commit prefetched registers into next buffer
        if (t + 1 < num_tiles) {
            __syncthreads();

            s_A[nxt][a_load_row][a_load_col + 0] = a_prefetch.x;
            s_A[nxt][a_load_row][a_load_col + 1] = a_prefetch.y;
            s_A[nxt][a_load_row][a_load_col + 2] = a_prefetch.z;
            s_A[nxt][a_load_row][a_load_col + 3] = a_prefetch.w;

            s_B[nxt][b_load_row][b_load_col + 0] = b_prefetch.x;
            s_B[nxt][b_load_row][b_load_col + 1] = b_prefetch.y;
            s_B[nxt][b_load_row][b_load_col + 2] = b_prefetch.z;
            s_B[nxt][b_load_row][b_load_col + 3] = b_prefetch.w;

            __syncthreads();
        }
    }

    // --- EPILOGUE: Vectorized 128-bit float4 Stores ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int g_row = cRow + threadRowInBlock + i;
            int g_col = cCol + threadColInBlock + j;

            if (g_row < M && g_col + 3 < N) {
                float4 c_out;
                if (beta == 0.0f) {
                    c_out.x = alpha * r_c[i][j + 0];
                    c_out.y = alpha * r_c[i][j + 1];
                    c_out.z = alpha * r_c[i][j + 2];
                    c_out.w = alpha * r_c[i][j + 3];
                } else {
                    float4 c_prev = *reinterpret_cast<const float4*>(&C[g_row * N + g_col]);
                    c_out.x = alpha * r_c[i][j + 0] + beta * c_prev.x;
                    c_out.y = alpha * r_c[i][j + 1] + beta * c_prev.y;
                    c_out.z = alpha * r_c[i][j + 2] + beta * c_prev.z;
                    c_out.w = alpha * r_c[i][j + 3] + beta * c_prev.w;
                }
                *reinterpret_cast<float4*>(&C[g_row * N + g_col]) = c_out;
            } else {
                for (int k = 0; k < 4; ++k) {
                    if (g_row < M && g_col + k < N) {
                        if (beta == 0.0f) {
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k];
                        } else {
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k] + beta * C[g_row * N + g_col + k];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 4. Split-K GEMM for Reduction-Heavy Shapes (K >> M, N)
// ============================================================================
__global__ void gemm_split_k_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K,
                                   int split_k,
                                   float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k_slice = blockIdx.z;

    int k_per_slice = (K + split_k - 1) / split_k;
    int k_start = k_slice * k_per_slice;
    int k_end = min(k_start + k_per_slice, K);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = k_start; k < k_end; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        atomicAdd(&C[row * N + col], alpha * sum);
    }
}

// ============================================================================
// 5. Transposed GEMMs (Backpropagation)
// ============================================================================

// C = A * B^T (A: M x K, B: N x K, C: M x N)
__global__ void gemm_NT_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               float alpha, float beta) {
    __shared__ float s_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float s_B[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_k = t * TILE_DIM + threadIdx.x;
        int b_k = t * TILE_DIM + threadIdx.y;

        if (row < M && a_k < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_k];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && b_k < K) {
            s_B[threadIdx.y][threadIdx.x] = B[col * K + b_k];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}

// C = A^T * B (A: K x M, B: K x N, C: M x N)
__global__ void gemm_TN_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               float alpha, float beta) {
    __shared__ float s_A[TILE_DIM][TILE_DIM + 1];
    __shared__ float s_B[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_k = t * TILE_DIM + threadIdx.x;
        int b_k = t * TILE_DIM + threadIdx.y;

        if (row < M && a_k < K) {
            s_A[threadIdx.y][threadIdx.x] = A[a_k * M + row];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && b_k < K) {
            s_B[threadIdx.y][threadIdx.x] = B[b_k * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta == 0.0f) {
            C[row * N + col] = alpha * acc;
        } else {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        }
    }
}

// ============================================================================
// Host Launchers & Shape-Aware Autotuning Dispatcher
// ============================================================================
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K,
                float alpha, float beta, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

void gemm_split_k(const float* A, const float* B, float* C, int M, int N, int K,
                  int split_k, float alpha, float beta, cudaStream_t stream) {
    if (beta == 0.0f) {
        CUDA_CHECK(cudaMemsetAsync(C, 0, M * N * sizeof(float), stream));
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, split_k);
    gemm_split_k_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, split_k, alpha, beta);
}

void gemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                         float alpha, float beta, cudaStream_t stream) {
    // Shape-Aware Dispatcher (Stage 6)
    if (K >= 2048 && (M * N <= 256 * 256)) {
        gemm_split_k(A, B, C, M, N, K, 8, alpha, beta, stream);
    } else if (M < 64 || N < 64) {
        gemm_tiled(A, B, C, M, N, K, alpha, beta, stream);
    } else {
        dim3 block((BM / TM) * (BN / TN)); // 256 threads arranged in warp hierarchy
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_double_buffered_warp_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void gemm_NT(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_NT_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

void gemm_TN(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_TN_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

} // namespace kernels
} // namespace cuda_ml

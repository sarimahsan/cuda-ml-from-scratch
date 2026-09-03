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
// 3. Medium-Matrix GEMM (BM=64, BN=64, BK=8, TM=4, TN=4, 256 Threads)
// Maximizes SM Occupancy for 256x256, 512x512, 1024x1024 matrices
// ============================================================================
__global__ void gemm_64x64_double_buffered_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int M, int N, int K,
                                                  float alpha, float beta) {
    __shared__ float s_A[2][64][9]; // +1 padding
    __shared__ float s_B[2][8][65];

    int cRow = blockIdx.y * 64;
    int cCol = blockIdx.x * 64;

    int threadRowInBlock = (threadIdx.x / 16) * 4;
    int threadColInBlock = (threadIdx.x % 16) * 4;

    float r_c[4][4] = {0.0f};

    int a_load_row = threadIdx.x / 4;
    int a_load_col = (threadIdx.x % 4) * 2;

    int b_load_row = threadIdx.x / 32;
    int b_load_col = (threadIdx.x % 32) * 2;

    auto load_tile = [&](int buffer_idx, int bk) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (cRow + a_load_row < M && bk + a_load_col + i < K) {
                s_A[buffer_idx][a_load_row][a_load_col + i] = A[(cRow + a_load_row) * K + (bk + a_load_col + i)];
            } else {
                s_A[buffer_idx][a_load_row][a_load_col + i] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (bk + b_load_row < K && cCol + b_load_col + i < N) {
                s_B[buffer_idx][b_load_row][b_load_col + i] = B[(bk + b_load_row) * N + (cCol + b_load_col + i)];
            } else {
                s_B[buffer_idx][b_load_row][b_load_col + i] = 0.0f;
            }
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int num_tiles = (K + 7) / 8;

    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = (t + 1) & 1;
        int next_bk = (t + 1) * 8;

        float2 a_pref = make_float2(0.0f, 0.0f);
        float2 b_pref = make_float2(0.0f, 0.0f);

        if (t + 1 < num_tiles) {
            if (cRow + a_load_row < M && next_bk + a_load_col < K) {
                a_pref.x = A[(cRow + a_load_row) * K + (next_bk + a_load_col + 0)];
                if (next_bk + a_load_col + 1 < K) {
                    a_pref.y = A[(cRow + a_load_row) * K + (next_bk + a_load_col + 1)];
                }
            }
            if (next_bk + b_load_row < K && cCol + b_load_col < N) {
                b_pref.x = B[(next_bk + b_load_row) * N + (cCol + b_load_col + 0)];
                if (cCol + b_load_col + 1 < N) {
                    b_pref.y = B[(next_bk + b_load_row) * N + (cCol + b_load_col + 1)];
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float r_a[4];
            float r_b[4];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                r_a[i] = s_A[cur][threadRowInBlock + i][k];
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                r_b[j] = s_B[cur][k][threadColInBlock + j];
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        if (t + 1 < num_tiles) {
            __syncthreads();
            s_A[nxt][a_load_row][a_load_col + 0] = a_pref.x;
            s_A[nxt][a_load_row][a_load_col + 1] = a_pref.y;

            s_B[nxt][b_load_row][b_load_col + 0] = b_pref.x;
            s_B[nxt][b_load_row][b_load_col + 1] = b_pref.y;
            __syncthreads();
        }
    }

    // Epilogue
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int g_row = cRow + threadRowInBlock + i;
            int g_col = cCol + threadColInBlock + j;
            if (g_row < M && g_col < N) {
                if (beta == 0.0f) {
                    C[g_row * N + g_col] = alpha * r_c[i][j];
                } else {
                    C[g_row * N + g_col] = alpha * r_c[i][j] + beta * C[g_row * N + g_col];
                }
            }
        }
    }
}

// ============================================================================
// 4. High-Throughput GEMM: Double Buffering + Warp Tiling + Vectorized Epilogue
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
// 5. Split-K GEMM using 128x128 Double-Buffered Warp-Tiled Engine
// Each K-slice computes a partial result with full arithmetic intensity,
// then accumulates into C via atomicAdd. Keeps 128x reuse per tile.
// ============================================================================
__global__ void gemm_split_k_double_buffered_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int M, int N, int K,
                                                   int split_k,
                                                   float alpha) {
    __shared__ float s_A[2][BM][BK + 1];
    __shared__ float s_B[2][BK][BN + 1];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    // K-slice range for this block (k_per_slice aligned to BK for float4 load alignment)
    int k_per_slice_raw = (K + split_k - 1) / split_k;
    int k_per_slice = ((k_per_slice_raw + BK - 1) / BK) * BK; // round up to BK=8
    int k_start = blockIdx.z * k_per_slice;
    int k_end = min(k_start + k_per_slice, K);
    int slice_K = k_end - k_start;
    if (slice_K <= 0) return;

    // Warp and Lane hierarchy (identical to base 128x128 kernel)
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = (warpId / 4) * 64;
    int warpCol = (warpId % 4) * 32;
    int laneRow = (laneId / 4) * TM;
    int laneCol = (laneId % 4) * TN;
    int threadRowInBlock = warpRow + laneRow;
    int threadColInBlock = warpCol + laneCol;

    float r_c[TM][TN] = {0.0f};

    int a_load_row = threadIdx.x / 2;
    int a_load_col = (threadIdx.x % 2) * 4;
    int b_load_row = threadIdx.x / 32;
    int b_load_col = (threadIdx.x % 32) * 4;

    // Lambda for loading a tile into shared memory (bounds-checked against k_end)
    auto load_tile_sk = [&](int buffer_idx, int bk) {
        if (cRow + a_load_row < M && bk + a_load_col + 3 < k_end) {
            float4 a_val = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (bk + a_load_col)]));
            s_A[buffer_idx][a_load_row][a_load_col + 0] = a_val.x;
            s_A[buffer_idx][a_load_row][a_load_col + 1] = a_val.y;
            s_A[buffer_idx][a_load_row][a_load_col + 2] = a_val.z;
            s_A[buffer_idx][a_load_row][a_load_col + 3] = a_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (cRow + a_load_row < M && bk + a_load_col + i < k_end)
                    s_A[buffer_idx][a_load_row][a_load_col + i] = A[(cRow + a_load_row) * K + (bk + a_load_col + i)];
                else
                    s_A[buffer_idx][a_load_row][a_load_col + i] = 0.0f;
            }
        }
        if (bk + b_load_row < k_end && cCol + b_load_col + 3 < N) {
            float4 b_val = __ldg(reinterpret_cast<const float4*>(&B[(bk + b_load_row) * N + (cCol + b_load_col)]));
            s_B[buffer_idx][b_load_row][b_load_col + 0] = b_val.x;
            s_B[buffer_idx][b_load_row][b_load_col + 1] = b_val.y;
            s_B[buffer_idx][b_load_row][b_load_col + 2] = b_val.z;
            s_B[buffer_idx][b_load_row][b_load_col + 3] = b_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (bk + b_load_row < k_end && cCol + b_load_col + i < N)
                    s_B[buffer_idx][b_load_row][b_load_col + i] = B[(bk + b_load_row) * N + (cCol + b_load_col + i)];
                else
                    s_B[buffer_idx][b_load_row][b_load_col + i] = 0.0f;
            }
        }
    };

    // Prologue: load first tile from k_start
    load_tile_sk(0, k_start);
    __syncthreads();

    int num_tiles = (slice_K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = (t + 1) & 1;
        int next_bk = k_start + (t + 1) * BK;

        float4 a_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (t + 1 < num_tiles) {
            if (cRow + a_load_row < M && next_bk + a_load_col + 3 < k_end) {
                a_prefetch = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (next_bk + a_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float v = 0.0f;
                    if (cRow + a_load_row < M && next_bk + a_load_col + i < k_end)
                        v = A[(cRow + a_load_row) * K + (next_bk + a_load_col + i)];
                    if (i == 0) a_prefetch.x = v;
                    else if (i == 1) a_prefetch.y = v;
                    else if (i == 2) a_prefetch.z = v;
                    else a_prefetch.w = v;
                }
            }
            if (next_bk + b_load_row < k_end && cCol + b_load_col + 3 < N) {
                b_prefetch = __ldg(reinterpret_cast<const float4*>(&B[(next_bk + b_load_row) * N + (cCol + b_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float v = 0.0f;
                    if (next_bk + b_load_row < k_end && cCol + b_load_col + i < N)
                        v = B[(next_bk + b_load_row) * N + (cCol + b_load_col + i)];
                    if (i == 0) b_prefetch.x = v;
                    else if (i == 1) b_prefetch.y = v;
                    else if (i == 2) b_prefetch.z = v;
                    else b_prefetch.w = v;
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float r_a[TM], r_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_a[i] = s_A[cur][threadRowInBlock + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_b[j] = s_B[cur][k][threadColInBlock + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    r_c[i][j] += r_a[i] * r_b[j];
        }

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

    // Epilogue: atomicAdd partial sums (only contention is between split_k slices)
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_row = cRow + threadRowInBlock + i;
            int g_col = cCol + threadColInBlock + j;
            if (g_row < M && g_col < N) {
                atomicAdd(&C[g_row * N + g_col], alpha * r_c[i][j]);
            }
        }
    }
}

// ============================================================================
// 6. Transposed GEMMs (Backpropagation)
// ============================================================================

// ============================================================================
// 6. C = A * B^T  (A: MxK row-major, B: NxK row-major, C: MxN row-major)
//    128x128 Double-Buffered Warp-Tiled — transposed B load
// ============================================================================
__global__ void gemm_NT_register_tiled_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K,
                                              float alpha, float beta) {
    __shared__ float s_A[2][BM][BK + 1];
    __shared__ float s_B[2][BK][BN + 1];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = (warpId / 4) * 64;
    int warpCol = (warpId % 4) * 32;
    int laneRow = (laneId / 4) * TM;
    int laneCol = (laneId % 4) * TN;
    int threadRowInBlock = warpRow + laneRow;
    int threadColInBlock = warpCol + laneCol;

    float r_c[TM][TN] = {0.0f};

    // A: MxK row-major — same load pattern as NN
    int a_load_row = threadIdx.x / 2;   // 0..127
    int a_load_col = (threadIdx.x % 2) * 4; // 0 or 4

    // B: NxK row-major (transposed) — load B^T tile: read BK rows of B^T = BK cols of B
    // We need s_B[k][n] = B[cCol+n, bk+k] = B[(cCol+n)*K + (bk+k)]
    // Remap: 256 threads load BK×BN = 8×128 elements
    // b_load_row = K-dim index (0..7), b_load_col = N-dim index within tile
    int b_load_row = threadIdx.x / 32;  // 0..7 (K-dim)
    int b_load_col = (threadIdx.x % 32) * 4; // 0,4,...,124 (N-dim)

    auto load_tile_nt = [&](int buffer_idx, int bk) {
        // A load: A[cRow+a_load_row, bk+a_load_col] (row-major)
        if (cRow + a_load_row < M && bk + a_load_col + 3 < K) {
            float4 a_val = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (bk + a_load_col)]));
            s_A[buffer_idx][a_load_row][a_load_col + 0] = a_val.x;
            s_A[buffer_idx][a_load_row][a_load_col + 1] = a_val.y;
            s_A[buffer_idx][a_load_row][a_load_col + 2] = a_val.z;
            s_A[buffer_idx][a_load_row][a_load_col + 3] = a_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (cRow + a_load_row < M && bk + a_load_col + i < K)
                    s_A[buffer_idx][a_load_row][a_load_col + i] = A[(cRow + a_load_row) * K + (bk + a_load_col + i)];
                else
                    s_A[buffer_idx][a_load_row][a_load_col + i] = 0.0f;
            }
        }

        // B^T load: s_B[k][n] = B[(cCol+n)*K + (bk+k)]
        // Each thread loads 4 consecutive N-elements for a single K-row
        // But B is NxK row-major, so B[n, k] = B[n*K + k]
        // For a fixed k (b_load_row), varying n: B[(cCol+b_load_col+i)*K + (bk+b_load_row)]
        // These are NOT contiguous in memory (stride = K), so we can't use float4 on B
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (bk + b_load_row < K && cCol + b_load_col + i < N)
                s_B[buffer_idx][b_load_row][b_load_col + i] = B[(cCol + b_load_col + i) * K + (bk + b_load_row)];
            else
                s_B[buffer_idx][b_load_row][b_load_col + i] = 0.0f;
        }
    };

    load_tile_nt(0, 0);
    __syncthreads();

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = (t + 1) & 1;
        int next_bk = (t + 1) * BK;

        // Prefetch A (same as NN)
        float4 a_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float b_pref[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (t + 1 < num_tiles) {
            if (cRow + a_load_row < M && next_bk + a_load_col + 3 < K) {
                a_prefetch = __ldg(reinterpret_cast<const float4*>(&A[(cRow + a_load_row) * K + (next_bk + a_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float v = 0.0f;
                    if (cRow + a_load_row < M && next_bk + a_load_col + i < K)
                        v = A[(cRow + a_load_row) * K + (next_bk + a_load_col + i)];
                    if (i == 0) a_prefetch.x = v;
                    else if (i == 1) a_prefetch.y = v;
                    else if (i == 2) a_prefetch.z = v;
                    else a_prefetch.w = v;
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (next_bk + b_load_row < K && cCol + b_load_col + i < N)
                    b_pref[i] = B[(cCol + b_load_col + i) * K + (next_bk + b_load_row)];
            }
        }

        // Compute
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float r_a[TM], r_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_a[i] = s_A[cur][threadRowInBlock + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_b[j] = s_B[cur][k][threadColInBlock + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    r_c[i][j] += r_a[i] * r_b[j];
        }

        if (t + 1 < num_tiles) {
            __syncthreads();
            s_A[nxt][a_load_row][a_load_col + 0] = a_prefetch.x;
            s_A[nxt][a_load_row][a_load_col + 1] = a_prefetch.y;
            s_A[nxt][a_load_row][a_load_col + 2] = a_prefetch.z;
            s_A[nxt][a_load_row][a_load_col + 3] = a_prefetch.w;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                s_B[nxt][b_load_row][b_load_col + i] = b_pref[i];
            __syncthreads();
        }
    }

    // Epilogue: float4 stores (same as NN)
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
                        if (beta == 0.0f)
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k];
                        else
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k] + beta * C[g_row * N + g_col + k];
                    }
                }
            }
        }
    }
}

// ============================================================================
// 7. C = A^T * B  (A: KxM col-major, B: KxN row-major, C: MxN row-major)
//    128x128 Double-Buffered Warp-Tiled — transposed A load
// ============================================================================
__global__ void gemm_TN_register_tiled_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K,
                                              float alpha, float beta) {
    __shared__ float s_A[2][BM][BK + 1];
    __shared__ float s_B[2][BK][BN + 1];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpRow = (warpId / 4) * 64;
    int warpCol = (warpId % 4) * 32;
    int laneRow = (laneId / 4) * TM;
    int laneCol = (laneId % 4) * TN;
    int threadRowInBlock = warpRow + laneRow;
    int threadColInBlock = warpCol + laneCol;

    float r_c[TM][TN] = {0.0f};

    // A: KxM col-major (A^T is MxK) — s_A[m][k] = A[k*M + (cRow+m)]
    // Load pattern: 256 threads load BM×BK = 128×8 elements
    // a_load_row = M-dim index (0..127), a_load_col = K-dim index
    // A^T[m, k] = A[k, m] = A[k * M + m] — NOT contiguous along K for fixed m
    int a_load_row = threadIdx.x / 2;   // 0..127 (M-dim)
    int a_load_col = (threadIdx.x % 2) * 4; // 0 or 4 (K-dim)

    // B: KxN row-major — same load pattern as NN
    int b_load_row = threadIdx.x / 32;  // 0..7 (K-dim)
    int b_load_col = (threadIdx.x % 32) * 4; // 0,4,...,124 (N-dim)

    auto load_tile_tn = [&](int buffer_idx, int bk) {
        // A^T load: s_A[a_load_row][a_load_col+i] = A[(bk+a_load_col+i)*M + (cRow+a_load_row)]
        // Stride-M access, can't vectorize with float4
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (cRow + a_load_row < M && bk + a_load_col + i < K)
                s_A[buffer_idx][a_load_row][a_load_col + i] = A[(bk + a_load_col + i) * M + (cRow + a_load_row)];
            else
                s_A[buffer_idx][a_load_row][a_load_col + i] = 0.0f;
        }

        // B load: B[bk+b_load_row, cCol+b_load_col] (row-major, contiguous along N)
        if (bk + b_load_row < K && cCol + b_load_col + 3 < N) {
            float4 b_val = __ldg(reinterpret_cast<const float4*>(&B[(bk + b_load_row) * N + (cCol + b_load_col)]));
            s_B[buffer_idx][b_load_row][b_load_col + 0] = b_val.x;
            s_B[buffer_idx][b_load_row][b_load_col + 1] = b_val.y;
            s_B[buffer_idx][b_load_row][b_load_col + 2] = b_val.z;
            s_B[buffer_idx][b_load_row][b_load_col + 3] = b_val.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (bk + b_load_row < K && cCol + b_load_col + i < N)
                    s_B[buffer_idx][b_load_row][b_load_col + i] = B[(bk + b_load_row) * N + (cCol + b_load_col + i)];
                else
                    s_B[buffer_idx][b_load_row][b_load_col + i] = 0.0f;
            }
        }
    };

    load_tile_tn(0, 0);
    __syncthreads();

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        int cur = t & 1;
        int nxt = (t + 1) & 1;
        int next_bk = (t + 1) * BK;

        // Prefetch
        float a_pref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 b_prefetch = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (t + 1 < num_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (cRow + a_load_row < M && next_bk + a_load_col + i < K)
                    a_pref[i] = A[(next_bk + a_load_col + i) * M + (cRow + a_load_row)];
            }
            if (next_bk + b_load_row < K && cCol + b_load_col + 3 < N) {
                b_prefetch = __ldg(reinterpret_cast<const float4*>(&B[(next_bk + b_load_row) * N + (cCol + b_load_col)]));
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float v = 0.0f;
                    if (next_bk + b_load_row < K && cCol + b_load_col + i < N)
                        v = B[(next_bk + b_load_row) * N + (cCol + b_load_col + i)];
                    if (i == 0) b_prefetch.x = v;
                    else if (i == 1) b_prefetch.y = v;
                    else if (i == 2) b_prefetch.z = v;
                    else b_prefetch.w = v;
                }
            }
        }

        // Compute (identical to NN)
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float r_a[TM], r_b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) r_a[i] = s_A[cur][threadRowInBlock + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) r_b[j] = s_B[cur][k][threadColInBlock + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    r_c[i][j] += r_a[i] * r_b[j];
        }

        if (t + 1 < num_tiles) {
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                s_A[nxt][a_load_row][a_load_col + i] = a_pref[i];
            s_B[nxt][b_load_row][b_load_col + 0] = b_prefetch.x;
            s_B[nxt][b_load_row][b_load_col + 1] = b_prefetch.y;
            s_B[nxt][b_load_row][b_load_col + 2] = b_prefetch.z;
            s_B[nxt][b_load_row][b_load_col + 3] = b_prefetch.w;
            __syncthreads();
        }
    }

    // Epilogue: float4 stores
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
                        if (beta == 0.0f)
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k];
                        else
                            C[g_row * N + g_col + k] = alpha * r_c[i][j + k] + beta * C[g_row * N + g_col + k];
                    }
                }
            }
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
    // Zero output buffer so atomicAdd accumulates correctly
    CUDA_CHECK(cudaMemsetAsync(C, 0, M * N * sizeof(float), stream));
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, split_k);
    gemm_split_k_double_buffered_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, split_k, alpha);
}

void gemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                         float alpha, float beta, cudaStream_t stream) {
    int blocks_128 = ((M + 127) / 128) * ((N + 127) / 128);

    // 1. Very small matrices: use 32x32 tiled
    if (M < 64 || N < 64) {
        gemm_tiled(A, B, C, M, N, K, alpha, beta, stream);
    }
    // 2. Occupancy-starved with large K: Split-K using full 128x128 engine
    //    e.g. 128x1024x4096 → blocks_128=8 → 8-way split → 64 blocks
    //    Keeps 128x arithmetic reuse, just slices the reduction dimension.
    else if (blocks_128 <= 16 && K >= 1024) {
        int target_blocks = 80; // ~2x SM count on Tesla T4 (40 SMs)
        int split_k = target_blocks / (blocks_128 > 0 ? blocks_128 : 1);
        if (split_k < 2) split_k = 2;
        if (split_k > 16) split_k = 16;
        gemm_split_k(A, B, C, M, N, K, split_k, alpha, beta, stream);
    }
    // 3. Tiny square matrices with short K: 64x64 for occupancy
    //    e.g. 256x256x256 → blocks_128=4, K=256 → 64x64 gives 16 blocks
    //    Only here because K is small enough that halved reuse doesn't hurt.
    else if (blocks_128 <= 4 && K <= 512) {
        dim3 block(256);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        gemm_64x64_double_buffered_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
    // 4. Default: 128x128 Double-Buffered Warp-Tiled (all other shapes)
    //    2048^3, 4096^3, 512^3, 1024^3, 4096x4096x64, etc.
    else {
        dim3 block(256);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_double_buffered_warp_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
}

void gemm_NT(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_NT_register_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

void gemm_TN(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_TN_register_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

} // namespace kernels
} // namespace cuda_ml

#include "../include/gemm.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"

namespace cuda_ml {
namespace kernels {

// 1. Naive GEMM: C = alpha * A * B + beta * C
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

// 2. 2D Shared Memory Tiled GEMM with bank conflict padding (TILE_DIM = 16 or 32)
#define TILE_DIM 16

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

// 3. Register Tiled GEMM: Each thread computes a 4x4 submatrix tile
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

__global__ void gemm_register_tiled_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K,
                                           float alpha, float beta) {
    __shared__ float s_A[BM][BK + 1];
    __shared__ float s_B[BK][BN + 1];

    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    int threadRow = (threadIdx.x / (BN / TN)) * TM;
    int threadCol = (threadIdx.x % (BN / TN)) * TN;

    float r_c[TM][TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        // Load s_A (BM x BK = 64 * 8 = 512 elements across 256 threads -> 2 elements per thread)
        #pragma unroll
        for (int offset = 0; offset < (BM * BK); offset += ((BM / TM) * (BN / TN))) {
            int idx = threadIdx.x + offset;
            int r = idx / BK;
            int c = idx % BK;
            if (cRow + r < M && bk + c < K) {
                s_A[r][c] = A[(cRow + r) * K + (bk + c)];
            } else {
                s_A[r][c] = 0.0f;
            }
        }

        // Load s_B (BK x BN = 8 * 64 = 512 elements across 256 threads -> 2 elements per thread)
        #pragma unroll
        for (int offset = 0; offset < (BK * BN); offset += ((BM / TM) * (BN / TN))) {
            int idx = threadIdx.x + offset;
            int r = idx / BN;
            int c = idx % BN;
            if (bk + r < K && cCol + c < N) {
                s_B[r][c] = B[(bk + r) * N + (cCol + c)];
            } else {
                s_B[r][c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float r_a[TM];
            float r_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_a[i] = s_A[threadRow + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_b[j] = s_B[k][threadCol + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_row = cRow + threadRow + i;
            int g_col = cCol + threadCol + j;
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

// 4. Transposed GEMMs
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
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_col = t * TILE_DIM + threadIdx.y; // note transposed B indexing

        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && b_col < K) {
            s_B[threadIdx.y][threadIdx.x] = B[col * K + b_col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[threadIdx.x][k];
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
        int a_row = t * TILE_DIM + threadIdx.y;
        int b_row = t * TILE_DIM + threadIdx.y;

        if (a_row < K && row < M) {
            s_A[threadIdx.y][threadIdx.x] = A[a_row * M + row];
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
            acc += s_A[k][threadIdx.y] * s_B[k][threadIdx.x];
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

// Host Launcher Implementations
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

void gemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K,
                         float alpha, float beta, cudaStream_t stream) {
    dim3 block((BM / TM) * (BN / TN)); // 256 threads per block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_register_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
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

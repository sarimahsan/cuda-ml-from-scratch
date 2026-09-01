#include "../include/linear.cuh"

// -----------------------------------------------------------------------------
// 2D Tiled Shared-Memory GEMMs for Linear Transformations and Projections
// Tile size: 16x16
// -----------------------------------------------------------------------------

#define TILE_DIM 16

// Out [M x N] = A [M x K] * B [K x N] + (bias != nullptr ? bias[col] : 0)
__global__ void gemm_forward_kernel(
    const float* __restrict__ d_A,
    const float* __restrict__ d_B,
    const float* __restrict__ d_bias,
    float* __restrict__ d_Out,
    int M,
    int K,
    int N
) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = d_A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = d_B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (d_bias != nullptr) {
            sum += d_bias[col];
        }
        d_Out[row * N + col] = sum;
    }
}

// dW [K x N] += A^T [K x M] * dY [M x N]
__global__ void gemm_backward_weights_kernel(
    const float* __restrict__ d_A,
    const float* __restrict__ d_dY,
    float* __restrict__ d_dW,
    int M,
    int K,
    int N,
    bool accumulate
) {
    __shared__ float s_AT[TILE_DIM][TILE_DIM];
    __shared__ float s_dY[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // in [0 .. K-1]
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // in [0 .. N-1]

    float sum = 0.0f;
    int num_tiles = (M + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_row = t * TILE_DIM + threadIdx.x;
        if (row < K && a_row < M) {
            s_AT[threadIdx.y][threadIdx.x] = d_A[a_row * K + row];
        } else {
            s_AT[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int dy_row = t * TILE_DIM + threadIdx.y;
        if (dy_row < M && col < N) {
            s_dY[threadIdx.y][threadIdx.x] = d_dY[dy_row * N + col];
        } else {
            s_dY[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_AT[threadIdx.y][k] * s_dY[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        if (accumulate) {
            d_dW[row * N + col] += sum;
        } else {
            d_dW[row * N + col] = sum;
        }
    }
}

// dX [M x K] = dY [M x N] * W^T [N x K]  (W is [K x N])
__global__ void gemm_backward_data_kernel(
    const float* __restrict__ d_dY,
    const float* __restrict__ d_W,
    float* __restrict__ d_dX,
    int M,
    int K,
    int N
) {
    __shared__ float s_dY[TILE_DIM][TILE_DIM];
    __shared__ float s_WT[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // in [0 .. M-1]
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // in [0 .. K-1]

    float sum = 0.0f;
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int dy_col = t * TILE_DIM + threadIdx.x;
        if (row < M && dy_col < N) {
            s_dY[threadIdx.y][threadIdx.x] = d_dY[row * N + dy_col];
        } else {
            s_dY[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int wt_row = t * TILE_DIM + threadIdx.y;
        if (wt_row < N && col < K) {
            // W is [K x N], so W^T at (wt_row, col) is W[col, wt_row]
            s_WT[threadIdx.y][threadIdx.x] = d_W[col * N + wt_row];
        } else {
            s_WT[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_dY[threadIdx.y][k] * s_WT[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        d_dX[row * K + col] = sum;
    }
}

// db [N] += sum_rows(dY [M x N])
__global__ void gemm_backward_bias_kernel(
    const float* __restrict__ d_dY,
    float* __restrict__ d_db,
    int M,
    int N,
    bool accumulate
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += d_dY[row * N + col];
        }
        if (accumulate) {
            d_db[col] += sum;
        } else {
            d_db[col] = sum;
        }
    }
}

void launch_gemm_forward(
    const float* d_A,
    const float* d_B,
    const float* d_bias,
    float* d_Out,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_forward_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_bias, d_Out, M, K, N);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_backward_weights(
    const float* d_A,
    const float* d_dY,
    float* d_dW,
    int M,
    int K,
    int N,
    bool accumulate,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM);
    gemm_backward_weights_kernel<<<grid, block, 0, stream>>>(d_A, d_dY, d_dW, M, K, N, accumulate);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_backward_data(
    const float* d_dY,
    const float* d_W,
    float* d_dX,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((K + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_backward_data_kernel<<<grid, block, 0, stream>>>(d_dY, d_W, d_dX, M, K, N);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_backward_bias(
    const float* d_dY,
    float* d_db,
    int M,
    int N,
    bool accumulate,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gemm_backward_bias_kernel<<<blocks, threads, 0, stream>>>(d_dY, d_db, M, N, accumulate);
    CUDA_KERNEL_CHECK();
}

#include "../include/linear.cuh"
#include <cuda_runtime.h>

#define TILE_DIM 16
#define WARP_SIZE 32

// -------------------------------------------------------------------------
// Warp & Block Reduction Helpers
// -------------------------------------------------------------------------
__inline__ __device__ float linear_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float linear_block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = linear_warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        sum = linear_warp_reduce_sum(sum);
    }
    return sum;
}

// -------------------------------------------------------------------------
// 1. Tiled GEMM Forward: Z = X * W + b
// X: [M x K], W: [K x N], b: [N], Z: [M x N]
// -------------------------------------------------------------------------
__global__ void tiled_gemm_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ Z,
    int M, int K, int N
) {
    __shared__ float s_X[TILE_DIM][TILE_DIM];
    __shared__ float s_W[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int k_x = t * TILE_DIM + threadIdx.x;
        int k_y = t * TILE_DIM + threadIdx.y;

        s_X[threadIdx.y][threadIdx.x] = (row < M && k_x < K) ? X[row * K + k_x] : 0.0f;
        s_W[threadIdx.y][threadIdx.x] = (k_y < K && col < N) ? W[k_y * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_X[threadIdx.y][i] * s_W[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float bias = (b != nullptr) ? b[col] : 0.0f;
        Z[row * N + col] = acc + bias;
    }
}

// -------------------------------------------------------------------------
// 2. Tiled GEMM Backward Gradients:
// dW = X^T * dZ   [K x N]
// dX = dZ * W^T   [M x K]
// db = sum_rows(dZ) [N]
// -------------------------------------------------------------------------
__global__ void tiled_gemm_backward_weight_kernel(
    const float* __restrict__ X,  // [M x K]
    const float* __restrict__ dZ, // [M x N]
    float* __restrict__ dW,       // [K x N]
    int M, int K, int N
) {
    __shared__ float s_X[TILE_DIM][TILE_DIM];
    __shared__ float s_dZ[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // index in K
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // index in N

    float acc = 0.0f;
    int num_tiles = (M + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int m_x = t * TILE_DIM + threadIdx.x; // sample index for X^T (row in X)
        int m_y = t * TILE_DIM + threadIdx.y; // sample index for dZ (row in dZ)

        s_X[threadIdx.y][threadIdx.x]  = (row < K && m_x < M) ? X[m_x * K + row] : 0.0f;
        s_dZ[threadIdx.y][threadIdx.x] = (m_y < M && col < N) ? dZ[m_y * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_X[threadIdx.y][i] * s_dZ[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        dW[row * N + col] = acc;
    }
}

__global__ void tiled_gemm_backward_input_kernel(
    const float* __restrict__ dZ, // [M x N]
    const float* __restrict__ W,  // [K x N]
    float* __restrict__ dX,       // [M x K]
    int M, int K, int N
) {
    __shared__ float s_dZ[TILE_DIM][TILE_DIM];
    __shared__ float s_W[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // index in M
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // index in K

    float acc = 0.0f;
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int n_x = t * TILE_DIM + threadIdx.x; // index in N
        int n_y = t * TILE_DIM + threadIdx.y; // index in N

        s_dZ[threadIdx.y][threadIdx.x] = (row < M && n_x < N) ? dZ[row * N + n_x] : 0.0f;
        s_W[threadIdx.y][threadIdx.x]  = (col < K && n_y < N) ? W[col * N + n_y] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += s_dZ[threadIdx.y][i] * s_W[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        dX[row * K + col] = acc;
    }
}

__global__ void bias_reduction_kernel(
    const float* __restrict__ dZ, // [M x N]
    float* __restrict__ db,       // [N]
    int M, int N
) {
    int col = blockIdx.x; // each block handles one column of dZ
    if (col >= N) return;

    float thread_sum = 0.0f;
    for (int row = threadIdx.x; row < M; row += blockDim.x) {
        thread_sum += dZ[row * N + col];
    }

    float block_sum = linear_block_reduce_sum(thread_sum);

    if (threadIdx.x == 0) {
        db[col] = block_sum;
    }
}

// -------------------------------------------------------------------------
// Host Launchers
// -------------------------------------------------------------------------
void linear_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_Z,
    int N, int D_in, int D_out,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((D_out + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    tiled_gemm_forward_kernel<<<grid, block, 0, stream>>>(d_X, d_W, d_b, d_Z, N, D_in, D_out);
    CUDA_CHECK_LAST();
}

void linear_backward(
    const float* d_dZ,
    const float* d_X,
    const float* d_W,
    float* d_dW,
    float* d_db,
    float* d_dX,
    int N, int D_in, int D_out,
    bool compute_dX,
    cudaStream_t stream
) {
    // 1. dW = X^T * dZ [D_in x D_out]
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid_dW((D_out + TILE_DIM - 1) / TILE_DIM, (D_in + TILE_DIM - 1) / TILE_DIM);
    tiled_gemm_backward_weight_kernel<<<grid_dW, block, 0, stream>>>(d_X, d_dZ, d_dW, N, D_in, D_out);
    CUDA_CHECK_LAST();

    // 2. db = sum_rows(dZ) [D_out]
    if (d_db != nullptr) {
        bias_reduction_kernel<<<D_out, 256, 0, stream>>>(d_dZ, d_db, N, D_out);
        CUDA_CHECK_LAST();
    }

    // 3. dX = dZ * W^T [N x D_in]
    if (compute_dX && d_dX != nullptr) {
        dim3 grid_dX((D_in + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
        tiled_gemm_backward_input_kernel<<<grid_dX, block, 0, stream>>>(d_dZ, d_W, d_dX, N, D_in, D_out);
        CUDA_CHECK_LAST();
    }
}

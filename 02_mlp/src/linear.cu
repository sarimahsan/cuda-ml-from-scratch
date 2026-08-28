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
        int m_X  = t * TILE_DIM + threadIdx.x;
        int m_dZ = t * TILE_DIM + threadIdx.y;

        // X^T[row, m_X] = X[m_X, row]
        s_X[threadIdx.y][threadIdx.x] = (m_X < M && row < K) ? X[m_X * K + row] : 0.0f;
        s_dZ[threadIdx.y][threadIdx.x] = (m_dZ < M && col < N) ? dZ[m_dZ * N + col] : 0.0f;

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

__global__ void tiled_gemm_backward_data_kernel(
    const float* __restrict__ dZ, // [M x N]
    const float* __restrict__ W,  // [K x N]
    float* __restrict__ dX,       // [M x K]
    int M, int N, int K
) {
    __shared__ float s_dZ[TILE_DIM][TILE_DIM];
    __shared__ float s_W[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y; // index in M
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // index in K

    float acc = 0.0f;
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int n_dZ = t * TILE_DIM + threadIdx.x;
        int n_W  = t * TILE_DIM + threadIdx.y;

        s_dZ[threadIdx.y][threadIdx.x] = (row < M && n_dZ < N) ? dZ[row * N + n_dZ] : 0.0f;
        // W^T[n_W, col] = W[col, n_W]
        s_W[threadIdx.y][threadIdx.x] = (col < K && n_W < N) ? W[col * N + n_W] : 0.0f;

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

__global__ void bias_gradient_reduction_kernel(
    const float* __restrict__ dZ,
    float* __restrict__ db,
    int M, int N
) {
    int col = blockIdx.x; // one block per column
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    if (col < N) {
        for (int row = tid; row < M; row += stride) {
            sum += dZ[row * N + col];
        }
        sum = linear_block_reduce_sum(sum);
        if (tid == 0) {
            db[col] = sum;
        }
    }
}

// -------------------------------------------------------------------------
// Host Launch Wrappers
// -------------------------------------------------------------------------
void launch_linear_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_Z,
    int M, int K, int N,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    tiled_gemm_forward_kernel<<<grid, block, 0, stream>>>(d_X, d_W, d_b, d_Z, M, K, N);
}

void launch_linear_backward(
    const float* d_dZ,
    const float* d_X,
    const float* d_W,
    float* d_dW,
    float* d_db,
    float* d_dX,
    int M, int K, int N,
    bool compute_dX,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);

    // 1. dW = X^T * dZ
    dim3 grid_dW((N + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM);
    tiled_gemm_backward_weight_kernel<<<grid_dW, block, 0, stream>>>(d_X, d_dZ, d_dW, M, K, N);

    // 2. db = sum_rows(dZ)
    bias_gradient_reduction_kernel<<<N, 256, 0, stream>>>(d_dZ, d_db, M, N);

    // 3. dX = dZ * W^T
    if (compute_dX && d_dX != nullptr) {
        dim3 grid_dX((K + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        tiled_gemm_backward_data_kernel<<<grid_dX, block, 0, stream>>>(d_dZ, d_W, d_dX, M, N, K);
    }
}

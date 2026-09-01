#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void gemm_forward_kernel_torch(
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

__global__ void gemm_backward_weights_kernel_torch(
    const float* __restrict__ d_A,
    const float* __restrict__ d_dY,
    float* __restrict__ d_dW,
    int M,
    int K,
    int N
) {
    __shared__ float s_AT[TILE_DIM][TILE_DIM];
    __shared__ float s_dY[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

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
        d_dW[row * N + col] = sum;
    }
}

__global__ void gemm_backward_data_kernel_torch(
    const float* __restrict__ d_dY,
    const float* __restrict__ d_W,
    float* __restrict__ d_dX,
    int M,
    int K,
    int N
) {
    __shared__ float s_dY[TILE_DIM][TILE_DIM];
    __shared__ float s_WT[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

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

__global__ void gemm_backward_bias_kernel_torch(
    const float* __restrict__ d_dY,
    float* __restrict__ d_db,
    int M,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += d_dY[row * N + col];
        }
        d_db[col] = sum;
    }
}

torch::Tensor linear_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto Out = torch::empty({M, N}, A.options());
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    gemm_forward_kernel_torch<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias_ptr, Out.data_ptr<float>(), M, K, N
    );
    return Out;
}

std::vector<torch::Tensor> linear_backward(
    torch::Tensor dY,
    torch::Tensor A,
    torch::Tensor B,
    bool compute_dX
) {
    TORCH_CHECK(dY.is_cuda() && A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto dW = torch::empty({K, N}, A.options());
    auto db = torch::empty({N}, A.options());

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid_w((N + TILE_DIM - 1) / TILE_DIM, (K + TILE_DIM - 1) / TILE_DIM);
    gemm_backward_weights_kernel_torch<<<grid_w, block>>>(
        A.data_ptr<float>(), dY.data_ptr<float>(), dW.data_ptr<float>(), M, K, N
    );

    int threads = 256;
    int blocks_b = (N + threads - 1) / threads;
    gemm_backward_bias_kernel_torch<<<blocks_b, threads>>>(
        dY.data_ptr<float>(), db.data_ptr<float>(), M, N
    );

    torch::Tensor dX;
    if (compute_dX) {
        dX = torch::empty({M, K}, A.options());
        dim3 grid_x((K + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        gemm_backward_data_kernel_torch<<<grid_x, block>>>(
            dY.data_ptr<float>(), B.data_ptr<float>(), dX.data_ptr<float>(), M, K, N
        );
    } else {
        dX = torch::empty({0}, A.options());
    }

    return {dW, db, dX};
}

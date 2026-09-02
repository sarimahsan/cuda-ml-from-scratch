#include "../include/pooling.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. MaxPool2D Forward Kernel
__global__ void maxpool2d_forward_kernel(const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int64_t* __restrict__ argmax_mask,
                                         int N, int C, int H_in, int W_in,
                                         int pool_h, int pool_w,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w,
                                         int H_out, int W_out) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z % C;
    int n     = blockIdx.z / C;

    if (n < N && c < C && h_out < H_out && w_out < W_out) {
        int h_start = h_out * stride_h - pad_h;
        int w_start = w_out * stride_w - pad_w;

        float max_val = -1e20f;
        int64_t max_idx = -1;

        const float* x_slice = X + (n * C + c) * H_in * W_in;

        for (int ph = 0; ph < pool_h; ++ph) {
            int h = h_start + ph;
            for (int pw = 0; pw < pool_w; ++pw) {
                int w = w_start + pw;
                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                    int in_idx = h * W_in + w;
                    float val = x_slice[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (n * C + c) * H_in * W_in + in_idx;
                    }
                }
            }
        }

        int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
        Y[out_idx] = max_val;
        if (argmax_mask) {
            argmax_mask[out_idx] = max_idx;
        }
    }
}

// MaxPool2D Backward Kernel
__global__ void maxpool2d_backward_kernel(const float* __restrict__ dY,
                                          const int64_t* __restrict__ argmax_mask,
                                          float* __restrict__ dX,
                                          int total_outputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_outputs) {
        int64_t in_idx = argmax_mask[idx];
        if (in_idx >= 0) {
            atomicAdd(&dX[in_idx], dY[idx]);
        }
    }
}

// 2. AvgPool2D Forward Kernel
__global__ void avgpool2d_forward_kernel(const float* __restrict__ X,
                                         float* __restrict__ Y,
                                         int N, int C, int H_in, int W_in,
                                         int pool_h, int pool_w,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w,
                                         int H_out, int W_out) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z % C;
    int n     = blockIdx.z / C;

    if (n < N && c < C && h_out < H_out && w_out < W_out) {
        int h_start = h_out * stride_h - pad_h;
        int w_start = w_out * stride_w - pad_w;

        float sum = 0.0f;
        int count = 0;

        const float* x_slice = X + (n * C + c) * H_in * W_in;

        for (int ph = 0; ph < pool_h; ++ph) {
            int h = h_start + ph;
            for (int pw = 0; pw < pool_w; ++pw) {
                int w = w_start + pw;
                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                    sum += x_slice[h * W_in + w];
                    count++;
                }
            }
        }

        int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
        Y[out_idx] = (count > 0) ? (sum / (float)count) : 0.0f;
    }
}

// Host Launchers
void maxpool2d_forward(const float* X, float* Y, int64_t* argmax_mask,
                       int N, int C, int H_in, int W_in,
                       int pool_h, int pool_w, int stride_h, int stride_w,
                       int pad_h, int pad_w, cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - pool_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - pool_w) / stride_w + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C);

    maxpool2d_forward_kernel<<<grid, block, 0, stream>>>(
        X, Y, argmax_mask, N, C, H_in, W_in,
        pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}

void maxpool2d_backward(const float* dY, const int64_t* argmax_mask, float* dX,
                        int N, int C, int H_in, int W_in,
                        int H_out, int W_out, cudaStream_t stream) {
    int total_in = N * C * H_in * W_in;
    CUDA_CHECK(cudaMemsetAsync(dX, 0, total_in * sizeof(float), stream));

    int total_outputs = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads, 0, stream>>>(
        dY, argmax_mask, dX, total_outputs);
}

void avgpool2d_forward(const float* X, float* Y,
                       int N, int C, int H_in, int W_in,
                       int pool_h, int pool_w, int stride_h, int stride_w,
                       int pad_h, int pad_w, cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - pool_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - pool_w) / stride_w + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C);

    avgpool2d_forward_kernel<<<grid, block, 0, stream>>>(
        X, Y, N, C, H_in, W_in, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}

} // namespace kernels
} // namespace cuda_ml

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// 2D Spatial Pooling: MaxPool2D (Packed Argmax Coordinates) & AvgPool2D
// X: (N, C, H_in, W_in), Y: (N, C, H_out, W_out)
// ============================================================================

// 1. MaxPool2D Forward: stores argmax indices for fast zero-search backward routing
void maxpool2d_forward(const float* X, float* Y, int64_t* argmax_mask,
                       int N, int C, int H_in, int W_in,
                       int pool_h, int pool_w, int stride_h, int stride_w,
                       int pad_h = 0, int pad_w = 0,
                       cudaStream_t stream = 0);

// MaxPool2D Backward: routes dY directly into dX using argmax_mask
void maxpool2d_backward(const float* dY, const int64_t* argmax_mask, float* dX,
                        int N, int C, int H_in, int W_in,
                        int H_out, int W_out,
                        cudaStream_t stream = 0);

// 2. AvgPool2D Forward & Backward
void avgpool2d_forward(const float* X, float* Y,
                       int N, int C, int H_in, int W_in,
                       int pool_h, int pool_w, int stride_h, int stride_w,
                       int pad_h = 0, int pad_w = 0,
                       cudaStream_t stream = 0);

void avgpool2d_backward(const float* dY, float* dX,
                        int N, int C, int H_in, int W_in,
                        int pool_h, int pool_w, int stride_h, int stride_w,
                        int pad_h = 0, int pad_w = 0,
                        cudaStream_t stream = 0);

// 3. Adaptive Global Average Pooling (maps H_in x W_in -> 1 x 1)
void global_avgpool2d_forward(const float* X, float* Y, int N, int C, int H, int W,
                              cudaStream_t stream = 0);

void global_avgpool2d_backward(const float* dY, float* dX, int N, int C, int H, int W,
                               cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

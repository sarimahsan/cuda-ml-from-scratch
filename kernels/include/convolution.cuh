#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// 2D Spatial & Im2Col Convolutions
// X: (N, C_in, H_in, W_in)
// W: (C_out, C_in, K_h, K_w)
// Bias: (C_out)
// Y: (N, C_out, H_out, W_out)
// ============================================================================

// 1. Direct 2D Spatial Forward Convolution with Shared Memory Halo Caching
void conv2d_direct_forward(const float* X, const float* W, const float* bias, float* Y,
                           int N, int C_in, int H_in, int W_in,
                           int C_out, int K_h, int K_w,
                           int pad_h, int pad_w, int stride_h, int stride_w,
                           cudaStream_t stream = 0);

// 2. Fused Conv2D + Bias + Activation (ReLU / GELU)
void conv2d_fused_forward(const float* X, const float* W, const float* bias, float* Y,
                          int N, int C_in, int H_in, int W_in,
                          int C_out, int K_h, int K_w,
                          int pad_h, int pad_w, int stride_h, int stride_w,
                          int activation_type, // 0: None, 1: ReLU, 2: GELU, 3: Sigmoid
                          cudaStream_t stream = 0);

// 3. Im2Col Transformation (Vectorized channel packing)
void im2col(const float* X, float* col,
            int N, int C_in, int H_in, int W_in,
            int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w,
            cudaStream_t stream = 0);

// 4. Col2Im Transformation (for backward gradient accumulation)
void col2im(const float* col, float* dX,
            int N, int C_in, int H_in, int W_in,
            int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w,
            cudaStream_t stream = 0);

// 5. Conv2D Backward Passes
void conv2d_backward_input(const float* dY, const float* W, float* dX,
                           int N, int C_in, int H_in, int W_in,
                           int C_out, int K_h, int K_w,
                           int pad_h, int pad_w, int stride_h, int stride_w,
                           cudaStream_t stream = 0);

void conv2d_backward_weight(const float* X, const float* dY, float* dW, float* dbias,
                            int N, int C_in, int H_in, int W_in,
                            int C_out, int K_h, int K_w,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

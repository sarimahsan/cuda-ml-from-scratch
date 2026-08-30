#ifndef CONV2D_CUH
#define CONV2D_CUH

#include "../../00_common/include/cuda_utils.cuh"

// Conv2D Forward: O = Conv2D(X, W) + b
// X: [N, C_in, H_in, W_in]
// W: [C_out, C_in, K_h, K_w]
// b: [C_out]
// O: [N, C_out, H_out, W_out]
void conv2d_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_O,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream = 0
);

// Conv2D Backward with respect to Input: dX = dO (padded, transposed) * W (flipped)
// dO: [N, C_out, H_out, W_out]
// W:  [C_out, C_in, K_h, K_w]
// dX: [N, C_in, H_in, W_in]
void conv2d_backward_data(
    const float* d_dO,
    const float* d_W,
    float* d_dX,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream = 0
);

// Conv2D Backward with respect to Weights & Bias:
// dW = X correlated with dO: dW[c_out, c_in, kh, kw] = sum_{n, h_out, w_out} dO * X
// db = sum_{n, h_out, w_out} dO[n, c_out, h_out, w_out]
void conv2d_backward_filter_bias(
    const float* d_X,
    const float* d_dO,
    float* d_dW,
    float* d_db,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream = 0
);

#endif // CONV2D_CUH

#ifndef POOL_CUH
#define POOL_CUH

#include "../../00_common/include/cuda_utils.cuh"

// MaxPool2D Forward: P = max_pool(X), storing argmax indices for backpropagation
// X: [N, C, H_in, W_in]
// P: [N, C, H_out, W_out]
// mask: [N, C, H_out, W_out] (stores flattened input index within input spatial dimensions)
void maxpool2d_forward(
    const float* d_X,
    float* d_P,
    int* d_mask,
    int N, int C, int H_in, int W_in,
    int pool_h, int pool_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream = 0
);

// MaxPool2D Backward: route dP back to the exact input index recorded in mask
// dP: [N, C, H_out, W_out]
// mask: [N, C, H_out, W_out]
// dX: [N, C, H_in, W_in] (initialized to 0)
void maxpool2d_backward(
    const float* d_dP,
    const int* d_mask,
    float* d_dX,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    cudaStream_t stream = 0
);

#endif // POOL_CUH

#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "../../00_common/include/cuda_utils.cuh"

// Linear Forward: Z = X * W + b
// X: [N, D_in], W: [D_in, D_out], b: [D_out], Z: [N, D_out]
void linear_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_Z,
    int N, int D_in, int D_out,
    cudaStream_t stream = 0
);

// Linear Backward:
// dW = X^T * dZ: [D_in, D_out]
// db = sum_rows(dZ): [D_out]
// dX = dZ * W^T: [N, D_in] (optional)
void linear_backward(
    const float* d_dZ,
    const float* d_X,
    const float* d_W,
    float* d_dW,
    float* d_db,
    float* d_dX,
    int N, int D_in, int D_out,
    bool compute_dX = true,
    cudaStream_t stream = 0
);

#endif // LINEAR_CUH

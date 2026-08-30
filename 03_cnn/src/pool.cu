#include "../include/pool.cuh"
#include <cfloat>
#include <cstdio>

// -------------------------------------------------------------------------
// MaxPool2D Forward Kernel
// -------------------------------------------------------------------------
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ P,
    int* __restrict__ mask,
    int N, int C, int H_in, int W_in,
    int pool_h, int pool_w,
    int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int rem = idx / W_out;
    int h_out = rem % H_out;
    rem = rem / H_out;
    int c = rem % C;
    int n = rem / C;

    int h_in_base = h_out * stride - pad;
    int w_in_base = w_out * stride - pad;

    int x_channel_offset = (n * C + c) * (H_in * W_in);

    float max_val = -FLT_MAX;
    int max_idx = -1;

    for (int ph = 0; ph < pool_h; ++ph) {
        int h_in = h_in_base + ph;
        if (h_in >= 0 && h_in < H_in) {
            int x_row_offset = x_channel_offset + h_in * W_in;

            for (int pw = 0; pw < pool_w; ++pw) {
                int w_in = w_in_base + pw;
                if (w_in >= 0 && w_in < W_in) {
                    int curr_idx = x_row_offset + w_in;
                    float val = X[curr_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = curr_idx;
                    }
                }
            }
        }
    }

    P[idx] = max_val;
    mask[idx] = max_idx;
}

// -------------------------------------------------------------------------
// MaxPool2D Backward Kernel
// -------------------------------------------------------------------------
__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ dP,
    const int* __restrict__ mask,
    float* __restrict__ dX,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int max_idx = mask[idx];
    if (max_idx >= 0) {
        atomicAdd(&dX[max_idx], dP[idx]);
    }
}

// -------------------------------------------------------------------------
// C++ Host Launchers
// -------------------------------------------------------------------------
void maxpool2d_forward(
    const float* d_X,
    float* d_P,
    int* d_mask,
    int N, int C, int H_in, int W_in,
    int pool_h, int pool_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total_elements = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    maxpool2d_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_X, d_P, d_mask,
        N, C, H_in, W_in,
        pool_h, pool_w,
        stride, pad,
        H_out, W_out
    );
    CUDA_CHECK_LAST();
}

void maxpool2d_backward(
    const float* d_dP,
    const int* d_mask,
    float* d_dX,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total_input_elements = N * C * H_in * W_in;
    cudaMemsetAsync(d_dX, 0, total_input_elements * sizeof(float), stream);

    int total_output_elements = N * C * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;

    maxpool2d_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_dP, d_mask, d_dX, total_output_elements
    );
    CUDA_CHECK_LAST();
}

#include "../include/conv2d.cuh"
#include <cstdio>

// -------------------------------------------------------------------------
// Conv2D Forward Kernel
// -------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ O,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int rem = idx / W_out;
    int h_out = rem % H_out;
    rem = rem / H_out;
    int c_out = rem % C_out;
    int n = rem / C_out;

    float acc = (b != nullptr) ? b[c_out] : 0.0f;

    int h_in_base = h_out * stride - pad;
    int w_in_base = w_out * stride - pad;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_channel_offset = (n * C_in + c_in) * (H_in * W_in);
        int w_channel_offset = ((c_out * C_in + c_in) * K_h) * K_w;

        for (int kh = 0; kh < K_h; ++kh) {
            int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < H_in) {
                int x_row_offset = x_channel_offset + h_in * W_in;
                int w_row_offset = w_channel_offset + kh * K_w;

                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        acc += X[x_row_offset + w_in] * W[w_row_offset + kw];
                    }
                }
            }
        }
    }

    O[idx] = acc;
}

// -------------------------------------------------------------------------
// Conv2D Backward Data (dX) Kernel
// -------------------------------------------------------------------------
__global__ void conv2d_backward_data_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ W,
    float* __restrict__ dX,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_in * H_in * W_in;
    if (idx >= total_elements) return;

    int w_in = idx % W_in;
    int rem = idx / W_in;
    int h_in = rem % H_in;
    rem = rem / H_in;
    int c_in = rem % C_in;
    int n = rem / C_in;

    float acc = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        int do_channel_offset = (n * C_out + c_out) * (H_out * W_out);
        int w_channel_offset = ((c_out * C_in + c_in) * K_h) * K_w;

        for (int kh = 0; kh < K_h; ++kh) {
            int h_out_num = h_in + pad - kh;
            if (h_out_num % stride == 0) {
                int h_out = h_out_num / stride;
                if (h_out >= 0 && h_out < H_out) {
                    int do_row_offset = do_channel_offset + h_out * W_out;
                    int w_row_offset = w_channel_offset + kh * K_w;

                    for (int kw = 0; kw < K_w; ++kw) {
                        int w_out_num = w_in + pad - kw;
                        if (w_out_num % stride == 0) {
                            int w_out = w_out_num / stride;
                            if (w_out >= 0 && w_out < W_out) {
                                acc += dO[do_row_offset + w_out] * W[w_row_offset + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    dX[idx] = acc;
}

// -------------------------------------------------------------------------
// Conv2D Backward Filter (dW) Kernel
// -------------------------------------------------------------------------
__global__ void conv2d_backward_filter_kernel(
    const float* __restrict__ X,
    const float* __restrict__ dO,
    float* __restrict__ dW,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = C_out * C_in * K_h * K_w;
    if (idx >= total_weights) return;

    int kw = idx % K_w;
    int rem = idx / K_w;
    int kh = rem % K_h;
    rem = rem / K_h;
    int c_in = rem % C_in;
    int c_out = rem / C_in;

    float acc = 0.0f;

    for (int n = 0; n < N; ++n) {
        int x_channel_offset = (n * C_in + c_in) * (H_in * W_in);
        int do_channel_offset = (n * C_out + c_out) * (H_out * W_out);

        for (int h_out = 0; h_out < H_out; ++h_out) {
            int h_in = h_out * stride - pad + kh;
            if (h_in >= 0 && h_in < H_in) {
                int x_row_offset = x_channel_offset + h_in * W_in;
                int do_row_offset = do_channel_offset + h_out * W_out;

                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int w_in = w_out * stride - pad + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        acc += dO[do_row_offset + w_out] * X[x_row_offset + w_in];
                    }
                }
            }
        }
    }

    dW[idx] = acc;
}

// -------------------------------------------------------------------------
// Conv2D Backward Bias (db) Kernel
// -------------------------------------------------------------------------
__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ dO,
    float* __restrict__ db,
    int N, int C_out, int H_out, int W_out
) {
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_out >= C_out) return;

    float acc = 0.0f;
    int spatial_size = H_out * W_out;

    for (int n = 0; n < N; ++n) {
        int offset = (n * C_out + c_out) * spatial_size;
        for (int i = 0; i < spatial_size; ++i) {
            acc += dO[offset + i];
        }
    }

    db[c_out] = acc;
}

// -------------------------------------------------------------------------
// C++ Host Launchers
// -------------------------------------------------------------------------
void conv2d_forward(
    const float* d_X,
    const float* d_W,
    const float* d_b,
    float* d_O,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total_elements = N * C_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    conv2d_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        d_X, d_W, d_b, d_O,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride, pad,
        H_out, W_out
    );
    CUDA_CHECK_LAST();
}

void conv2d_backward_data(
    const float* d_dO,
    const float* d_W,
    float* d_dX,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total_elements = N * C_in * H_in * W_in;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    conv2d_backward_data_kernel<<<grid_size, block_size, 0, stream>>>(
        d_dO, d_W, d_dX,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride, pad,
        H_out, W_out
    );
    CUDA_CHECK_LAST();
}

void conv2d_backward_filter_bias(
    const float* d_X,
    const float* d_dO,
    float* d_dW,
    float* d_db,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total_weights = C_out * C_in * K_h * K_w;
    int block_size = 256;
    int grid_size = (total_weights + block_size - 1) / block_size;

    conv2d_backward_filter_kernel<<<grid_size, block_size, 0, stream>>>(
        d_X, d_dO, d_dW,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride, pad,
        H_out, W_out
    );
    CUDA_CHECK_LAST();

    if (d_db != nullptr) {
        int bias_grid = (C_out + block_size - 1) / block_size;
        conv2d_backward_bias_kernel<<<bias_grid, block_size, 0, stream>>>(
            d_dO, d_db, N, C_out, H_out, W_out
        );
        CUDA_CHECK_LAST();
    }
}

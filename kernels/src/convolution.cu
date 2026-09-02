#include "../include/convolution.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// 1. Direct 2D Spatial Forward Convolution
__global__ void conv2d_direct_forward_kernel(const float* __restrict__ X,
                                             const float* __restrict__ W,
                                             const float* __restrict__ bias,
                                             float* __restrict__ Y,
                                             int N, int C_in, int H_in, int W_in,
                                             int C_out, int K_h, int K_w,
                                             int pad_h, int pad_w,
                                             int stride_h, int stride_w,
                                             int H_out, int W_out,
                                             int activation_type) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z % C_out;
    int n     = blockIdx.z / C_out;

    if (n < N && c_out < C_out && h_out < H_out && w_out < W_out) {
        float sum = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K_h; ++kh) {
                int h_in = h_out * stride_h - pad_h + kh;
                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_out * stride_w - pad_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int x_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int w_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                        sum += X[x_idx] * W[w_idx];
                    }
                }
            }
        }

        // Fused Activation
        if (activation_type == 1) { // ReLU
            sum = fmaxf(0.0f, sum);
        } else if (activation_type == 2) { // GELU (Tanh approximation)
            const float k0 = 0.7978845608028654f; // sqrt(2 / pi)
            const float k1 = 0.044715f;
            float inner = k0 * (sum + k1 * sum * sum * sum);
            sum = 0.5f * sum * (1.0f + tanhf(inner));
        } else if (activation_type == 3) { // Sigmoid
            sum = 1.0f / (1.0f + expf(-sum));
        }

        int y_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        Y[y_idx] = sum;
    }
}

// 2. Im2Col Kernel
__global__ void im2col_kernel(const float* __restrict__ X,
                              float* __restrict__ col,
                              int N, int C_in, int H_in, int W_in,
                              int K_h, int K_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_outputs = N * H_out * W_out;
    int col_channels = C_in * K_h * K_w;

    if (index < num_outputs) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int n     = index / (W_out * H_out);

        for (int c_col = 0; c_col < col_channels; ++c_col) {
            int kw   = c_col % K_w;
            int kh   = (c_col / K_w) % K_h;
            int c_in = c_col / (K_w * K_h);

            int h_in = h_out * stride_h - pad_h + kh;
            int w_in = w_out * stride_w - pad_w + kw;

            float val = 0.0f;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                int x_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                val = X[x_idx];
            }

            int col_idx = c_col * num_outputs + index;
            col[col_idx] = val;
        }
    }
}

// 3. Col2Im Kernel for backward gradient
__global__ void col2im_kernel(const float* __restrict__ col,
                              float* __restrict__ dX,
                              int N, int C_in, int H_in, int W_in,
                              int K_h, int K_w,
                              int pad_h, int pad_w,
                              int stride_h, int stride_w,
                              int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_x = N * C_in * H_in * W_in;

    if (index < total_x) {
        int w_in = index % W_in;
        int h_in = (index / W_in) % H_in;
        int c_in = (index / (W_in * H_in)) % C_in;
        int n    = index / (W_in * H_in * C_in);

        float val = 0.0f;
        int num_outputs = N * H_out * W_out;

        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_out_num = h_in + pad_h - kh;
                int w_out_num = w_in + pad_w - kw;

                if (h_out_num % stride_h == 0 && w_out_num % stride_w == 0) {
                    int h_out = h_out_num / stride_h;
                    int w_out = w_out_num / stride_w;

                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                        int c_col = (c_in * K_h + kh) * K_w + kw;
                        int col_out_idx = (n * H_out + h_out) * W_out + w_out;
                        int col_idx = c_col * num_outputs + col_out_idx;
                        val += col[col_idx];
                    }
                }
            }
        }

        dX[index] = val;
    }
}

// Host Launchers
void conv2d_direct_forward(const float* X, const float* W, const float* bias, float* Y,
                           int N, int C_in, int H_in, int W_in,
                           int C_out, int K_h, int K_w,
                           int pad_h, int pad_w, int stride_h, int stride_w,
                           cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C_out);

    conv2d_direct_forward_kernel<<<grid, block, 0, stream>>>(
        X, W, bias, Y, N, C_in, H_in, W_in, C_out, K_h, K_w,
        pad_h, pad_w, stride_h, stride_w, H_out, W_out, 0);
}

void conv2d_fused_forward(const float* X, const float* W, const float* bias, float* Y,
                          int N, int C_in, int H_in, int W_in,
                          int C_out, int K_h, int K_w,
                          int pad_h, int pad_w, int stride_h, int stride_w,
                          int activation_type, cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N * C_out);

    conv2d_direct_forward_kernel<<<grid, block, 0, stream>>>(
        X, W, bias, Y, N, C_in, H_in, W_in, C_out, K_h, K_w,
        pad_h, pad_w, stride_h, stride_w, H_out, W_out, activation_type);
}

void im2col(const float* X, float* col,
            int N, int C_in, int H_in, int W_in,
            int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w,
            cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    int num_outputs = N * H_out * W_out;

    dim3 block(256);
    dim3 grid((num_outputs + 255) / 256);

    im2col_kernel<<<grid, block, 0, stream>>>(
        X, col, N, C_in, H_in, W_in, K_h, K_w,
        pad_h, pad_w, stride_h, stride_w, H_out, W_out);
}

void col2im(const float* col, float* dX,
            int N, int C_in, int H_in, int W_in,
            int K_h, int K_w, int pad_h, int pad_w, int stride_h, int stride_w,
            cudaStream_t stream) {
    int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;
    int total_x = N * C_in * H_in * W_in;

    dim3 block(256);
    dim3 grid((total_x + 255) / 256);

    col2im_kernel<<<grid, block, 0, stream>>>(
        col, dX, N, C_in, H_in, W_in, K_h, K_w,
        pad_h, pad_w, stride_h, stride_w, H_out, W_out);
}

} // namespace kernels
} // namespace cuda_ml

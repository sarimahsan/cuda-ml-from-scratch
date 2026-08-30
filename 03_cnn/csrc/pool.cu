#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

// -------------------------------------------------------------------------
// MaxPool2D Forward Kernel
// -------------------------------------------------------------------------
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ P,
    int64_t* __restrict__ mask,
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
    int64_t max_idx = -1;

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
    const int64_t* __restrict__ mask,
    float* __restrict__ dX,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t max_idx = mask[idx];
    if (max_idx >= 0) {
        atomicAdd(&dX[max_idx], dP[idx]);
    }
}

// -------------------------------------------------------------------------
// PyTorch Extension Wrappers
// -------------------------------------------------------------------------
std::vector<torch::Tensor> maxpool2d_forward_cuda(
    torch::Tensor X,
    int pool_h,
    int pool_w,
    int stride,
    int pad
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    int N = X.size(0);
    int C = X.size(1);
    int H_in = X.size(2);
    int W_in = X.size(3);

    int H_out = (H_in + 2 * pad - pool_h) / stride + 1;
    int W_out = (W_in + 2 * pad - pool_w) / stride + 1;

    auto P = torch::empty({N, C, H_out, W_out}, X.options());
    auto mask = torch::empty({N, C, H_out, W_out}, X.options().dtype(torch::kInt64));

    int total_elements = N * C * H_out * W_out;
    int block = 256;
    int grid = (total_elements + block - 1) / block;

    maxpool2d_forward_kernel<<<grid, block>>>(
        X.data_ptr<float>(),
        P.data_ptr<float>(),
        mask.data_ptr<int64_t>(),
        N, C, H_in, W_in,
        pool_h, pool_w,
        stride, pad,
        H_out, W_out
    );

    return {P, mask};
}

torch::Tensor maxpool2d_backward_cuda(
    torch::Tensor dP,
    torch::Tensor mask,
    int N, int C, int H_in, int W_in
) {
    TORCH_CHECK(dP.is_cuda() && mask.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dP.is_contiguous() && mask.is_contiguous(), "Inputs must be contiguous");

    auto dX = torch::zeros({N, C, H_in, W_in}, dP.options());

    int total_elements = dP.numel();
    int block = 256;
    int grid = (total_elements + block - 1) / block;

    maxpool2d_backward_kernel<<<grid, block>>>(
        dP.data_ptr<float>(),
        mask.data_ptr<int64_t>(),
        dX.data_ptr<float>(),
        total_elements
    );

    return dX;
}

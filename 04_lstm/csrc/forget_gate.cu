#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void forget_gate_forward_kernel_torch(
    const float* __restrict__ d_Z_f,
    float* __restrict__ d_f,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_f[idx];
        d_f[idx] = sigmoidf_dev(z);
    }
}

__global__ void forget_gate_backward_kernel_torch(
    const float* __restrict__ d_df,
    const float* __restrict__ d_f,
    float* __restrict__ d_dZ_f,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float df = d_df[idx];
        float f_val = d_f[idx];
        d_dZ_f[idx] = df * f_val * (1.0f - f_val);
    }
}

torch::Tensor forget_gate_forward(torch::Tensor Z_f) {
    TORCH_CHECK(Z_f.is_cuda(), "Z_f must be a CUDA tensor");
    TORCH_CHECK(Z_f.is_contiguous(), "Z_f must be contiguous");

    auto out = torch::empty_like(Z_f);
    int size = Z_f.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    forget_gate_forward_kernel_torch<<<blocks, threads>>>(
        Z_f.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}

torch::Tensor forget_gate_backward(torch::Tensor df, torch::Tensor f_val) {
    TORCH_CHECK(df.is_cuda() && f_val.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(df.is_contiguous() && f_val.is_contiguous(), "Inputs must be contiguous");

    auto dZ_f = torch::empty_like(df);
    int size = df.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    forget_gate_backward_kernel_torch<<<blocks, threads>>>(
        df.data_ptr<float>(), f_val.data_ptr<float>(), dZ_f.data_ptr<float>(), size
    );
    return dZ_f;
}

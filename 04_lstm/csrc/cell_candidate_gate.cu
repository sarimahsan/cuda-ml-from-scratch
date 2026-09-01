#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void candidate_gate_forward_kernel_torch(
    const float* __restrict__ d_Z_g,
    float* __restrict__ d_g,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_g[idx];
        d_g[idx] = tanhf(z);
    }
}

__global__ void candidate_gate_backward_kernel_torch(
    const float* __restrict__ d_dg,
    const float* __restrict__ d_g,
    float* __restrict__ d_dZ_g,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dg = d_dg[idx];
        float g_val = d_g[idx];
        d_dZ_g[idx] = dg * (1.0f - g_val * g_val);
    }
}

torch::Tensor candidate_gate_forward(torch::Tensor Z_g) {
    TORCH_CHECK(Z_g.is_cuda(), "Z_g must be a CUDA tensor");
    TORCH_CHECK(Z_g.is_contiguous(), "Z_g must be contiguous");

    auto out = torch::empty_like(Z_g);
    int size = Z_g.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    candidate_gate_forward_kernel_torch<<<blocks, threads>>>(
        Z_g.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}

torch::Tensor candidate_gate_backward(torch::Tensor dg, torch::Tensor g_val) {
    TORCH_CHECK(dg.is_cuda() && g_val.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dg.is_contiguous() && g_val.is_contiguous(), "Inputs must be contiguous");

    auto dZ_g = torch::empty_like(dg);
    int size = dg.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    candidate_gate_backward_kernel_torch<<<blocks, threads>>>(
        dg.data_ptr<float>(), g_val.data_ptr<float>(), dZ_g.data_ptr<float>(), size
    );
    return dZ_g;
}

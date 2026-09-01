#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void output_gate_forward_kernel_torch(
    const float* __restrict__ d_Z_o,
    float* __restrict__ d_o,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_o[idx];
        d_o[idx] = sigmoidf_dev(z);
    }
}

__global__ void output_gate_backward_kernel_torch(
    const float* __restrict__ d_do,
    const float* __restrict__ d_o,
    float* __restrict__ d_dZ_o,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float do_val = d_do[idx];
        float o_val = d_o[idx];
        d_dZ_o[idx] = do_val * o_val * (1.0f - o_val);
    }
}

torch::Tensor output_gate_forward(torch::Tensor Z_o) {
    TORCH_CHECK(Z_o.is_cuda(), "Z_o must be a CUDA tensor");
    TORCH_CHECK(Z_o.is_contiguous(), "Z_o must be contiguous");

    auto out = torch::empty_like(Z_o);
    int size = Z_o.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    output_gate_forward_kernel_torch<<<blocks, threads>>>(
        Z_o.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}

torch::Tensor output_gate_backward(torch::Tensor do_t, torch::Tensor o_val) {
    TORCH_CHECK(do_t.is_cuda() && o_val.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(do_t.is_contiguous() && o_val.is_contiguous(), "Inputs must be contiguous");

    auto dZ_o = torch::empty_like(do_t);
    int size = do_t.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    output_gate_backward_kernel_torch<<<blocks, threads>>>(
        do_t.data_ptr<float>(), o_val.data_ptr<float>(), dZ_o.data_ptr<float>(), size
    );
    return dZ_o;
}

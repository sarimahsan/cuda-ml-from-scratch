#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void input_gate_forward_kernel_torch(
    const float* __restrict__ d_Z_i,
    float* __restrict__ d_i,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = d_Z_i[idx];
        d_i[idx] = sigmoidf_dev(z);
    }
}

__global__ void input_gate_backward_kernel_torch(
    const float* __restrict__ d_di,
    const float* __restrict__ d_i,
    float* __restrict__ d_dZ_i,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float di = d_di[idx];
        float i_val = d_i[idx];
        d_dZ_i[idx] = di * i_val * (1.0f - i_val);
    }
}

torch::Tensor input_gate_forward(torch::Tensor Z_i) {
    TORCH_CHECK(Z_i.is_cuda(), "Z_i must be a CUDA tensor");
    TORCH_CHECK(Z_i.is_contiguous(), "Z_i must be contiguous");

    auto out = torch::empty_like(Z_i);
    int size = Z_i.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    input_gate_forward_kernel_torch<<<blocks, threads>>>(
        Z_i.data_ptr<float>(), out.data_ptr<float>(), size
    );
    return out;
}

torch::Tensor input_gate_backward(torch::Tensor di, torch::Tensor i_val) {
    TORCH_CHECK(di.is_cuda() && i_val.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(di.is_contiguous() && i_val.is_contiguous(), "Inputs must be contiguous");

    auto dZ_i = torch::empty_like(di);
    int size = di.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    input_gate_backward_kernel_torch<<<blocks, threads>>>(
        di.data_ptr<float>(), i_val.data_ptr<float>(), dZ_i.data_ptr<float>(), size
    );
    return dZ_i;
}

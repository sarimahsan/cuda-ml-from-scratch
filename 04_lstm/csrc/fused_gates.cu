#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void fused_lstm_gates_forward_kernel_torch(
    const float* __restrict__ d_gates_preact, // [N x 4H]
    const float* __restrict__ d_c_prev,       // [N x H]
    float* __restrict__ d_gates_act,          // [N x 4H] -> [i, f, g, o]
    float* __restrict__ d_c_next,             // [N x H]
    float* __restrict__ d_tanh_c,             // [N x H]
    float* __restrict__ d_h_next,             // [N x H]
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_4h = n * (4 * H);
        float z_i = d_gates_preact[base_4h + h];
        float z_f = d_gates_preact[base_4h + H + h];
        float z_g = d_gates_preact[base_4h + 2 * H + h];
        float z_o = d_gates_preact[base_4h + 3 * H + h];

        float i_val = sigmoidf_dev(z_i);
        float f_val = sigmoidf_dev(z_f);
        float g_val = tanhf(z_g);
        float o_val = sigmoidf_dev(z_o);

        if (d_gates_act) {
            d_gates_act[base_4h + h] = i_val;
            d_gates_act[base_4h + H + h] = f_val;
            d_gates_act[base_4h + 2 * H + h] = g_val;
            d_gates_act[base_4h + 3 * H + h] = o_val;
        }

        float c_prev = d_c_prev ? d_c_prev[idx] : 0.0f;
        float c_next = f_val * c_prev + i_val * g_val;
        float tc = tanhf(c_next);
        float h_next = o_val * tc;

        d_c_next[idx] = c_next;
        if (d_tanh_c) d_tanh_c[idx] = tc;
        d_h_next[idx] = h_next;
    }
}

__global__ void fused_lstm_gates_backward_kernel_torch(
    const float* __restrict__ d_dh,           // [N x H]
    const float* __restrict__ d_dc_next,      // [N x H]
    const float* __restrict__ d_gates_act,    // [N x 4H] -> [i, f, g, o]
    const float* __restrict__ d_c_prev,       // [N x H]
    const float* __restrict__ d_c_next,       // [N x H]
    const float* __restrict__ d_tanh_c,       // [N x H]
    float* __restrict__ d_dgates_preact,      // [N x 4H] -> [dZ_i, dZ_f, dZ_g, dZ_o]
    float* __restrict__ d_dc_prev,            // [N x H]
    int N,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * H;
    if (idx < total_elements) {
        int n = idx / H;
        int h = idx % H;

        int base_4h = n * (4 * H);
        float i_val = d_gates_act[base_4h + h];
        float f_val = d_gates_act[base_4h + H + h];
        float g_val = d_gates_act[base_4h + 2 * H + h];
        float o_val = d_gates_act[base_4h + 3 * H + h];

        float dh = d_dh[idx];
        float dc_next = d_dc_next ? d_dc_next[idx] : 0.0f;
        float tc = d_tanh_c ? d_tanh_c[idx] : tanhf(d_c_next[idx]);
        float c_prev = d_c_prev ? d_c_prev[idx] : 0.0f;

        float do_val = dh * tc;
        float dz_o = do_val * o_val * (1.0f - o_val);

        float dtanh_c = dh * o_val;
        float dc_total = dtanh_c * (1.0f - tc * tc) + dc_next;

        float dc_prev_val = dc_total * f_val;
        if (d_dc_prev) d_dc_prev[idx] = dc_prev_val;

        float df_val = dc_total * c_prev;
        float dz_f = df_val * f_val * (1.0f - f_val);

        float di_val = dc_total * g_val;
        float dz_i = di_val * i_val * (1.0f - i_val);

        float dg_val = dc_total * i_val;
        float dz_g = dg_val * (1.0f - g_val * g_val);

        d_dgates_preact[base_4h + h] = dz_i;
        d_dgates_preact[base_4h + H + h] = dz_f;
        d_dgates_preact[base_4h + 2 * H + h] = dz_g;
        d_dgates_preact[base_4h + 3 * H + h] = dz_o;
    }
}

std::vector<torch::Tensor> fused_lstm_gates_forward(
    torch::Tensor gates_preact,
    torch::Tensor c_prev
) {
    TORCH_CHECK(gates_preact.is_cuda(), "gates_preact must be CUDA tensor");
    int N = gates_preact.size(0);
    int four_H = gates_preact.size(1);
    int H = four_H / 4;

    auto gates_act = torch::empty_like(gates_preact);
    auto c_next = torch::empty({N, H}, gates_preact.options());
    auto tanh_c = torch::empty({N, H}, gates_preact.options());
    auto h_next = torch::empty({N, H}, gates_preact.options());

    int total = N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* c_prev_ptr = c_prev.defined() ? c_prev.data_ptr<float>() : nullptr;

    fused_lstm_gates_forward_kernel_torch<<<blocks, threads>>>(
        gates_preact.data_ptr<float>(),
        c_prev_ptr,
        gates_act.data_ptr<float>(),
        c_next.data_ptr<float>(),
        tanh_c.data_ptr<float>(),
        h_next.data_ptr<float>(),
        N, H
    );

    return {h_next, c_next, gates_act, tanh_c};
}

std::vector<torch::Tensor> fused_lstm_gates_backward(
    torch::Tensor dh,
    torch::Tensor dc_next,
    torch::Tensor gates_act,
    torch::Tensor c_prev,
    torch::Tensor c_next,
    torch::Tensor tanh_c
) {
    TORCH_CHECK(dh.is_cuda() && gates_act.is_cuda(), "Inputs must be CUDA tensors");
    int N = dh.size(0);
    int H = dh.size(1);

    auto dgates_preact = torch::empty_like(gates_act);
    auto dc_prev = torch::empty({N, H}, dh.options());

    int total = N * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* dc_next_ptr = dc_next.defined() ? dc_next.data_ptr<float>() : nullptr;
    const float* c_prev_ptr = c_prev.defined() ? c_prev.data_ptr<float>() : nullptr;
    const float* c_next_ptr = c_next.defined() ? c_next.data_ptr<float>() : nullptr;
    const float* tanh_c_ptr = tanh_c.defined() ? tanh_c.data_ptr<float>() : nullptr;

    fused_lstm_gates_backward_kernel_torch<<<blocks, threads>>>(
        dh.data_ptr<float>(),
        dc_next_ptr,
        gates_act.data_ptr<float>(),
        c_prev_ptr,
        c_next_ptr,
        tanh_c_ptr,
        dgates_preact.data_ptr<float>(),
        dc_prev.data_ptr<float>(),
        N, H
    );

    return {dgates_preact, dc_prev};
}

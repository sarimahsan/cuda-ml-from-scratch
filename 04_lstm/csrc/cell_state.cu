#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cell_state_forward_kernel_torch(
    const float* __restrict__ d_f,
    const float* __restrict__ d_c_prev,
    const float* __restrict__ d_i,
    const float* __restrict__ d_g,
    const float* __restrict__ d_o,
    float* __restrict__ d_c,
    float* __restrict__ d_tanh_c,
    float* __restrict__ d_h,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float f_val = d_f[idx];
        float c_prev_val = d_c_prev ? d_c_prev[idx] : 0.0f;
        float i_val = d_i[idx];
        float g_val = d_g[idx];
        float o_val = d_o[idx];

        float c_val = f_val * c_prev_val + i_val * g_val;
        float tc_val = tanhf(c_val);
        float h_val = o_val * tc_val;

        d_c[idx] = c_val;
        if (d_tanh_c) d_tanh_c[idx] = tc_val;
        d_h[idx] = h_val;
    }
}

__global__ void cell_state_backward_kernel_torch(
    const float* __restrict__ d_dh,
    const float* __restrict__ d_dc_next,
    const float* __restrict__ d_o,
    const float* __restrict__ d_tanh_c,
    const float* __restrict__ d_c_prev,
    const float* __restrict__ d_f,
    const float* __restrict__ d_i,
    const float* __restrict__ d_g,
    float* __restrict__ d_dc_prev,
    float* __restrict__ d_df,
    float* __restrict__ d_di,
    float* __restrict__ d_dg,
    float* __restrict__ d_do,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dh = d_dh[idx];
        float dc_next = d_dc_next ? d_dc_next[idx] : 0.0f;
        float o_val = d_o[idx];
        float tc_val = d_tanh_c[idx];
        float c_prev_val = d_c_prev ? d_c_prev[idx] : 0.0f;
        float f_val = d_f[idx];
        float i_val = d_i[idx];
        float g_val = d_g[idx];

        // 1. do_t = dh_t * tanh(c_t)
        float do_val = dh * tc_val;

        // 2. dc_t_total = dh_t * o_t * (1 - tanh_c^2) + dc_next
        float dtanh_c = dh * o_val;
        float dc_total = dtanh_c * (1.0f - tc_val * tc_val) + dc_next;

        // 3. dc_prev = dc_total * f_t
        float dc_prev_val = dc_total * f_val;

        // 4. df_t = dc_total * c_{t-1}
        float df_val = dc_total * c_prev_val;

        // 5. di_t = dc_total * g_t
        float di_val = dc_total * g_val;

        // 6. dg_t = dc_total * i_t
        float dg_val = dc_total * i_val;

        if (d_do) d_do[idx] = do_val;
        if (d_dc_prev) d_dc_prev[idx] = dc_prev_val;
        if (d_df) d_df[idx] = df_val;
        if (d_di) d_di[idx] = di_val;
        if (d_dg) d_dg[idx] = dg_val;
    }
}

std::vector<torch::Tensor> cell_state_forward(
    torch::Tensor f_val,
    torch::Tensor c_prev,
    torch::Tensor i_val,
    torch::Tensor g_val,
    torch::Tensor o_val
) {
    auto c_next = torch::empty_like(f_val);
    auto tanh_c = torch::empty_like(f_val);
    auto h_next = torch::empty_like(f_val);

    int size = f_val.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    const float* c_prev_ptr = c_prev.defined() ? c_prev.data_ptr<float>() : nullptr;

    cell_state_forward_kernel_torch<<<blocks, threads>>>(
        f_val.data_ptr<float>(),
        c_prev_ptr,
        i_val.data_ptr<float>(),
        g_val.data_ptr<float>(),
        o_val.data_ptr<float>(),
        c_next.data_ptr<float>(),
        tanh_c.data_ptr<float>(),
        h_next.data_ptr<float>(),
        size
    );

    return {c_next, tanh_c, h_next};
}

std::vector<torch::Tensor> cell_state_backward(
    torch::Tensor dh,
    torch::Tensor dc_next,
    torch::Tensor o_val,
    torch::Tensor tanh_c,
    torch::Tensor c_prev,
    torch::Tensor f_val,
    torch::Tensor i_val,
    torch::Tensor g_val
) {
    auto dc_prev = torch::empty_like(f_val);
    auto df = torch::empty_like(f_val);
    auto di = torch::empty_like(f_val);
    auto dg = torch::empty_like(f_val);
    auto do_t = torch::empty_like(f_val);

    int size = f_val.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    const float* dc_next_ptr = dc_next.defined() ? dc_next.data_ptr<float>() : nullptr;
    const float* c_prev_ptr = c_prev.defined() ? c_prev.data_ptr<float>() : nullptr;

    cell_state_backward_kernel_torch<<<blocks, threads>>>(
        dh.data_ptr<float>(),
        dc_next_ptr,
        o_val.data_ptr<float>(),
        tanh_c.data_ptr<float>(),
        c_prev_ptr,
        f_val.data_ptr<float>(),
        i_val.data_ptr<float>(),
        g_val.data_ptr<float>(),
        dc_prev.data_ptr<float>(),
        df.data_ptr<float>(),
        di.data_ptr<float>(),
        dg.data_ptr<float>(),
        do_t.data_ptr<float>(),
        size
    );

    return {dc_prev, df, di, dg, do_t};
}

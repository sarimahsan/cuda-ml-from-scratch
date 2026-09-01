#include "../include/lstm.cuh"
#include <random>
#include <cstring>
#include <cmath>
#include <iostream>

CUDALSTM::CUDALSTM(const LSTMConfig& config) : cfg(config), opt_step(0) {
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;
    int T = cfg.seq_len;
    int N = cfg.batch_size;

    // Weights:
    // W_ih: [D x 4H] (transposed layout for fast GEMM: X [N x D] * W_ih [D x 4H] = [N x 4H])
    // W_hh: [H x 4H] (H_prev [N x H] * W_hh [H x 4H] = [N x 4H])
    // W_out: [H x V] (H_t [N x H] * W_out [H x V] = [N x V])
    d_W_ih = allocate_device_memory<float>(D * 4 * H);
    d_b_ih = allocate_device_memory<float>(4 * H);
    d_W_hh = allocate_device_memory<float>(H * 4 * H);
    d_b_hh = allocate_device_memory<float>(4 * H);
    d_W_out = allocate_device_memory<float>(H * V);
    d_b_out = allocate_device_memory<float>(V);

    // Gradients:
    d_dW_ih = allocate_device_memory<float>(D * 4 * H);
    d_db_ih = allocate_device_memory<float>(4 * H);
    d_dW_hh = allocate_device_memory<float>(H * 4 * H);
    d_db_hh = allocate_device_memory<float>(4 * H);
    d_dW_out = allocate_device_memory<float>(H * V);
    d_db_out = allocate_device_memory<float>(V);

    // Optimizer states:
    d_m_W_ih = allocate_device_memory<float>(D * 4 * H);
    d_v_W_ih = allocate_device_memory<float>(D * 4 * H);
    d_m_b_ih = allocate_device_memory<float>(4 * H);
    d_v_b_ih = allocate_device_memory<float>(4 * H);

    d_m_W_hh = allocate_device_memory<float>(H * 4 * H);
    d_v_W_hh = allocate_device_memory<float>(H * 4 * H);
    d_m_b_hh = allocate_device_memory<float>(4 * H);
    d_v_b_hh = allocate_device_memory<float>(4 * H);

    d_m_W_out = allocate_device_memory<float>(H * V);
    d_v_W_out = allocate_device_memory<float>(H * V);
    d_m_b_out = allocate_device_memory<float>(V);
    d_v_b_out = allocate_device_memory<float>(V);

    // Sequence States & Pre-activations:
    d_X_seq = allocate_device_memory<float>(T * N * D);
    d_G_ih = allocate_device_memory<float>(T * N * 4 * H);
    d_G_hh = allocate_device_memory<float>(T * N * 4 * H);
    d_G_total = allocate_device_memory<float>(T * N * 4 * H);
    d_G_act = allocate_device_memory<float>(T * N * 4 * H);
    d_C_seq = allocate_device_memory<float>((T + 1) * N * H);
    d_Tanh_C_seq = allocate_device_memory<float>(T * N * H);
    d_H_seq = allocate_device_memory<float>((T + 1) * N * H);
    d_Logits_seq = allocate_device_memory<float>(T * N * V);
    d_Probs_seq = allocate_device_memory<float>(T * N * V);
    d_Losses_seq = allocate_device_memory<float>(T * N);
    d_Targets_seq = allocate_device_memory<int>(T * N);

    // Backward buffers:
    d_dLogits_seq = allocate_device_memory<float>(T * N * V);
    d_dH_seq = allocate_device_memory<float>((T + 1) * N * H);
    d_dC_seq = allocate_device_memory<float>((T + 1) * N * H);
    d_dG_seq = allocate_device_memory<float>(T * N * 4 * H);
    d_dX_seq = allocate_device_memory<float>(T * N * D);

    reset_states();
}

CUDALSTM::~CUDALSTM() {
    free_device_memory(d_W_ih);
    free_device_memory(d_b_ih);
    free_device_memory(d_W_hh);
    free_device_memory(d_b_hh);
    free_device_memory(d_W_out);
    free_device_memory(d_b_out);

    free_device_memory(d_dW_ih);
    free_device_memory(d_db_ih);
    free_device_memory(d_dW_hh);
    free_device_memory(d_db_hh);
    free_device_memory(d_dW_out);
    free_device_memory(d_db_out);

    free_device_memory(d_m_W_ih); free_device_memory(d_v_W_ih);
    free_device_memory(d_m_b_ih); free_device_memory(d_v_b_ih);
    free_device_memory(d_m_W_hh); free_device_memory(d_v_W_hh);
    free_device_memory(d_m_b_hh); free_device_memory(d_v_b_hh);
    free_device_memory(d_m_W_out); free_device_memory(d_v_W_out);
    free_device_memory(d_m_b_out); free_device_memory(d_v_b_out);

    free_device_memory(d_X_seq);
    free_device_memory(d_G_ih);
    free_device_memory(d_G_hh);
    free_device_memory(d_G_total);
    free_device_memory(d_G_act);
    free_device_memory(d_C_seq);
    free_device_memory(d_Tanh_C_seq);
    free_device_memory(d_H_seq);
    free_device_memory(d_Logits_seq);
    free_device_memory(d_Probs_seq);
    free_device_memory(d_Losses_seq);
    free_device_memory(d_Targets_seq);

    free_device_memory(d_dLogits_seq);
    free_device_memory(d_dH_seq);
    free_device_memory(d_dC_seq);
    free_device_memory(d_dG_seq);
    free_device_memory(d_dX_seq);
}

void CUDALSTM::reset_states() {
    int H = cfg.hidden_dim;
    int N = cfg.batch_size;
    int T = cfg.seq_len;
    // Zero out initial hidden and cell state (C[0] and H[0])
    CUDA_CHECK(cudaMemset(d_C_seq, 0, (T + 1) * N * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_H_seq, 0, (T + 1) * N * H * sizeof(float)));

    // Zero out optimizer states
    int D = cfg.input_dim;
    int V = cfg.vocab_size;
    CUDA_CHECK(cudaMemset(d_m_W_ih, 0, D * 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_ih, 0, D * 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b_ih, 0, 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_ih, 0, 4 * H * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_W_hh, 0, H * 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_hh, 0, H * 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b_hh, 0, 4 * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_hh, 0, 4 * H * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_W_out, 0, H * V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_out, 0, H * V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b_out, 0, V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_out, 0, V * sizeof(float)));
}

void CUDALSTM::init_weights(unsigned int seed) {
    std::mt19937 gen(seed);
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;

    // Xavier / Glorot uniform initialization: [-a, a] with a = sqrt(6 / (fan_in + fan_out))
    auto init_tensor = [&](float* d_ptr, int fan_in, int fan_out, int size) {
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> h_buf(size);
        for (int i = 0; i < size; ++i) h_buf[i] = dist(gen);
        CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    };

    init_tensor(d_W_ih, D, 4 * H, D * 4 * H);
    init_tensor(d_W_hh, H, 4 * H, H * 4 * H);
    init_tensor(d_W_out, H, V, H * V);

    // Biases initialized to 0, except forget gate bias initialized to 1.0 (recommended for LSTM stability)
    std::vector<float> h_b_ih(4 * H, 0.0f);
    std::vector<float> h_b_hh(4 * H, 0.0f);
    for (int h = 0; h < H; ++h) {
        h_b_ih[H + h] = 1.0f; // forget gate bias
    }
    CUDA_CHECK(cudaMemcpy(d_b_ih, h_b_ih.data(), 4 * H * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_hh, h_b_hh.data(), 4 * H * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_b_out(V, 0.0f);
    CUDA_CHECK(cudaMemcpy(d_b_out, h_b_out.data(), V * sizeof(float), cudaMemcpyHostToDevice));
}

// Helper kernel to combine pre-activations: G_total = G_ih + G_hh + b_ih + b_hh
__global__ void add_gates_and_biases_kernel(
    const float* __restrict__ d_G_ih,
    const float* __restrict__ d_G_hh,
    const float* __restrict__ d_b_ih,
    const float* __restrict__ d_b_hh,
    float* __restrict__ d_G_total,
    int N,
    int four_H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * four_H;
    if (idx < total) {
        int col = idx % four_H;
        float val = d_G_ih[idx] + d_G_hh[idx];
        if (d_b_ih) val += d_b_ih[col];
        if (d_b_hh) val += d_b_hh[col];
        d_G_total[idx] = val;
    }
}

float CUDALSTM::forward(const float* h_X_seq, const int* h_targets_seq, cudaStream_t stream) {
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;
    int T = cfg.seq_len;
    int N = cfg.batch_size;

    // Copy batch sequence data to device
    if (h_X_seq) {
        CUDA_CHECK(cudaMemcpyAsync(d_X_seq, h_X_seq, T * N * D * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    if (h_targets_seq) {
        CUDA_CHECK(cudaMemcpyAsync(d_Targets_seq, h_targets_seq, T * N * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    // Unroll forward pass through sequence timesteps t = 0 .. T-1
    for (int t = 0; t < T; ++t) {
        float* d_xt = d_X_seq + t * N * D;
        float* d_g_ih_t = d_G_ih + t * N * 4 * H;
        float* d_g_hh_t = d_G_hh + t * N * 4 * H;
        float* d_g_tot_t = d_G_total + t * N * 4 * H;
        float* d_g_act_t = d_G_act + t * N * 4 * H;

        float* d_c_prev = d_C_seq + t * N * H;
        float* d_c_next = d_C_seq + (t + 1) * N * H;
        float* d_tanh_c_t = d_Tanh_C_seq + t * N * H;

        float* d_h_prev = d_H_seq + t * N * H;
        float* d_h_next = d_H_seq + (t + 1) * N * H;

        float* d_logits_t = d_Logits_seq + t * N * V;

        // 1. GEMM 1: G_ih = X_t [N x D] * W_ih [D x 4H]
        launch_gemm_forward(d_xt, d_W_ih, nullptr, d_g_ih_t, N, D, 4 * H, stream);

        // 2. GEMM 2: G_hh = H_{t-1} [N x H] * W_hh [H x 4H]
        launch_gemm_forward(d_h_prev, d_W_hh, nullptr, d_g_hh_t, N, H, 4 * H, stream);

        // 3. Add: G_total = G_ih + G_hh + b_ih + b_hh
        int total_gates = N * 4 * H;
        int threads = 256;
        int blocks = (total_gates + threads - 1) / threads;
        add_gates_and_biases_kernel<<<blocks, threads, 0, stream>>>(
            d_g_ih_t, d_g_hh_t, d_b_ih, d_b_hh, d_g_tot_t, N, 4 * H
        );

        // 4. Gates & Cell State calculation
        if (cfg.use_fused_gate) {
            // Fused 4-Gate Forward Kernel
            launch_fused_lstm_gates_forward(
                d_g_tot_t, d_c_prev, d_g_act_t, d_c_next, d_tanh_c_t, d_h_next, N, H, stream
            );
        } else {
            // Modular Separate Gates execution
            float* d_i_t = d_g_act_t;
            float* d_f_t = d_g_act_t + N * H;
            float* d_g_t = d_g_act_t + 2 * N * H;
            float* d_o_t = d_g_act_t + 3 * N * H;

            launch_input_gate_forward(d_g_tot_t, d_i_t, N * H, stream);
            launch_forget_gate_forward(d_g_tot_t + N * H, d_f_t, N * H, stream);
            launch_candidate_gate_forward(d_g_tot_t + 2 * N * H, d_g_t, N * H, stream);
            launch_output_gate_forward(d_g_tot_t + 3 * N * H, d_o_t, N * H, stream);

            launch_cell_state_forward(
                d_f_t, d_c_prev, d_i_t, d_g_t, d_o_t, d_c_next, d_tanh_c_t, d_h_next, N * H, stream
            );
        }

        // 5. Output Projection: Logits_t [N x V] = H_t [N x H] * W_out [H x V] + b_out
        launch_gemm_forward(d_h_next, d_W_out, d_b_out, d_logits_t, N, H, V, stream);
    }

    // 6. Sequence Softmax & Cross-Entropy Loss
    int total_tokens = T * N;
    launch_sequence_softmax_cross_entropy(
        d_Logits_seq, d_Targets_seq, d_Probs_seq, d_Losses_seq, d_dLogits_seq, total_tokens, V, stream
    );

    return compute_mean_loss(d_Losses_seq, total_tokens, stream);
}

void CUDALSTM::backward(cudaStream_t stream) {
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;
    int T = cfg.seq_len;
    int N = cfg.batch_size;

    // Reset gradient accumulators
    CUDA_CHECK(cudaMemsetAsync(d_dW_ih, 0, D * 4 * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_db_ih, 0, 4 * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dW_hh, 0, H * 4 * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_db_hh, 0, 4 * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dW_out, 0, H * V * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_db_out, 0, V * sizeof(float), stream));

    // Zero out terminal recurrent gradient buffers (dh_{T+1} = 0, dc_{T+1} = 0)
    CUDA_CHECK(cudaMemsetAsync(d_dH_seq, 0, (T + 1) * N * H * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dC_seq, 0, (T + 1) * N * H * sizeof(float), stream));

    // Backpropagation Through Time (BPTT): t = T-1 down to 0
    for (int t = T - 1; t >= 0; --t) {
        float* d_xt = d_X_seq + t * N * D;
        float* d_h_prev = d_H_seq + t * N * H;
        float* d_h_curr = d_H_seq + (t + 1) * N * H;
        float* d_c_prev = d_C_seq + t * N * H;
        float* d_c_curr = d_C_seq + (t + 1) * N * H;
        float* d_tanh_c = d_Tanh_C_seq + t * N * H;

        float* d_g_act_t = d_G_act + t * N * 4 * H;
        float* d_dlogits_t = d_dLogits_seq + t * N * V;

        float* d_dh_curr = d_dH_seq + (t + 1) * N * H;
        float* d_dh_prev = d_dH_seq + t * N * H;
        float* d_dc_curr = d_dC_seq + (t + 1) * N * H;
        float* d_dc_prev = d_dC_seq + t * N * H;

        float* d_dg_tot_t = d_dG_seq + t * N * 4 * H;
        float* d_dxt = d_dX_seq + t * N * D;

        // 1. Output Linear layer backward:
        // dW_out += H_curr^T [H x N] * dLogits_t [N x V]
        launch_gemm_backward_weights(d_h_curr, d_dlogits_t, d_dW_out, N, H, V, true, stream);
        // db_out += sum(dLogits_t)
        launch_gemm_backward_bias(d_dlogits_t, d_db_out, N, V, true, stream);
        // Add projection gradient to dh_curr: dH_curr += dLogits_t * W_out^T
        launch_gemm_backward_data(d_dlogits_t, d_W_out, d_dh_curr, N, H, V, stream);

        // 2. Gates Backward:
        if (cfg.use_fused_gate) {
            launch_fused_lstm_gates_backward(
                d_dh_curr, d_dc_curr, d_g_act_t, d_c_prev, d_c_curr, d_tanh_c,
                d_dg_tot_t, d_dc_prev, N, H, stream
            );
        } else {
            float* d_i_t = d_g_act_t;
            float* d_f_t = d_g_act_t + N * H;
            float* d_g_t = d_g_act_t + 2 * N * H;
            float* d_o_t = d_g_act_t + 3 * N * H;

            float* d_di_t = d_dg_tot_t;
            float* d_df_t = d_dg_tot_t + N * H;
            float* d_dg_t = d_dg_tot_t + 2 * N * H;
            float* d_do_t = d_dg_tot_t + 3 * N * H;

            launch_cell_state_backward(
                d_dh_curr, d_dc_curr, d_o_t, d_tanh_c, d_c_prev, d_f_t, d_i_t, d_g_t,
                d_dc_prev, d_df_t, d_di_t, d_dg_t, d_do_t, N * H, stream
            );

            launch_input_gate_backward(d_di_t, d_i_t, d_di_t, N * H, stream);
            launch_forget_gate_backward(d_df_t, d_f_t, d_df_t, N * H, stream);
            launch_candidate_gate_backward(d_dg_t, d_g_t, d_dg_t, N * H, stream);
            launch_output_gate_backward(d_do_t, d_o_t, d_do_t, N * H, stream);
        }

        // 3. Weight & Bias gradient accumulation:
        // dW_ih += X_t^T [D x N] * dG_total [N x 4H]
        launch_gemm_backward_weights(d_xt, d_dg_tot_t, d_dW_ih, N, D, 4 * H, true, stream);
        // db_ih += sum(dG_total)
        launch_gemm_backward_bias(d_dg_tot_t, d_db_ih, N, 4 * H, true, stream);

        // dW_hh += H_prev^T [H x N] * dG_total [N x 4H]
        launch_gemm_backward_weights(d_h_prev, d_dg_tot_t, d_dW_hh, N, H, 4 * H, true, stream);
        // db_hh += sum(dG_total)
        launch_gemm_backward_bias(d_dg_tot_t, d_db_hh, N, 4 * H, true, stream);

        // 4. Data gradient & recurrent hidden gradient:
        // dX_t = dG_total [N x 4H] * W_ih^T [4H x D]
        launch_gemm_backward_data(d_dg_tot_t, d_W_ih, d_dxt, N, D, 4 * H, stream);
        // dH_prev = dG_total [N x 4H] * W_hh^T [4H x H]
        launch_gemm_backward_data(d_dg_tot_t, d_W_hh, d_dh_prev, N, H, 4 * H, stream);
    }
}

void CUDALSTM::step(cudaStream_t stream) {
    opt_step++;
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;

    // Gradient norm clipping:
    float* grad_ptrs[6] = { d_dW_ih, d_db_ih, d_dW_hh, d_db_hh, d_dW_out, d_db_out };
    int grad_sizes[6] = { D * 4 * H, 4 * H, H * 4 * H, 4 * H, H * V, V };
    launch_clip_grad_norm(grad_ptrs, grad_sizes, 6, cfg.clip_norm, stream);

    // Adam optimizer update for each parameter:
    launch_adam_step(d_W_ih, d_m_W_ih, d_v_W_ih, d_dW_ih, D * 4 * H, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);
    launch_adam_step(d_b_ih, d_m_b_ih, d_v_b_ih, d_db_ih, 4 * H, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);

    launch_adam_step(d_W_hh, d_m_W_hh, d_v_W_hh, d_dW_hh, H * 4 * H, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);
    launch_adam_step(d_b_hh, d_m_b_hh, d_v_b_hh, d_db_hh, 4 * H, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);

    launch_adam_step(d_W_out, d_m_W_out, d_v_W_out, d_dW_out, H * V, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);
    launch_adam_step(d_b_out, d_m_b_out, d_v_b_out, d_db_out, V, cfg.learning_rate, 0.9f, 0.999f, 1e-8f, opt_step, stream);
}

void CUDALSTM::predict_step(
    const float* d_x_t,
    const float* d_h_prev,
    const float* d_c_prev,
    float* d_h_next,
    float* d_c_next,
    float* d_logits,
    cudaStream_t stream
) {
    int D = cfg.input_dim;
    int H = cfg.hidden_dim;
    int V = cfg.vocab_size;
    int N = 1; // single sample prediction

    float* d_g_ih = allocate_device_memory<float>(4 * H);
    float* d_g_hh = allocate_device_memory<float>(4 * H);
    float* d_g_tot = allocate_device_memory<float>(4 * H);
    float* d_g_act = allocate_device_memory<float>(4 * H);
    float* d_tanh_c = allocate_device_memory<float>(H);

    launch_gemm_forward(d_x_t, d_W_ih, nullptr, d_g_ih, 1, D, 4 * H, stream);
    launch_gemm_forward(d_h_prev, d_W_hh, nullptr, d_g_hh, 1, H, 4 * H, stream);

    int threads = 128;
    int blocks = (4 * H + threads - 1) / threads;
    add_gates_and_biases_kernel<<<blocks, threads, 0, stream>>>(
        d_g_ih, d_g_hh, d_b_ih, d_b_hh, d_g_tot, 1, 4 * H
    );

    launch_fused_lstm_gates_forward(
        d_g_tot, d_c_prev, d_g_act, d_c_next, d_tanh_c, d_h_next, 1, H, stream
    );

    launch_gemm_forward(d_h_next, d_W_out, d_b_out, d_logits, 1, H, V, stream);

    free_device_memory(d_g_ih);
    free_device_memory(d_g_hh);
    free_device_memory(d_g_tot);
    free_device_memory(d_g_act);
    free_device_memory(d_tanh_c);
}

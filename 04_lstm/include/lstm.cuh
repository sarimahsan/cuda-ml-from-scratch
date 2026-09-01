#pragma once

#include "../../00_common/include/cuda_utils.cuh"
#include "input_gate.cuh"
#include "forget_gate.cuh"
#include "cell_candidate_gate.cuh"
#include "output_gate.cuh"
#include "cell_state.cuh"
#include "fused_gates.cuh"
#include "linear.cuh"
#include "softmax_loss.cuh"
#include "optimizers.cuh"

#include <vector>
#include <string>

struct LSTMConfig {
    int input_dim;       // D (e.g. vocab_size or embedding_dim)
    int hidden_dim;      // H (e.g. 128 or 256)
    int vocab_size;      // V (for language modeling projection)
    int seq_len;         // T
    int batch_size;      // N
    float learning_rate; // eta
    float clip_norm;     // e.g. 1.0 or 5.0
    bool use_fused_gate; // whether to use fused kernel or separate gate kernels
};

class CUDALSTM {
public:
    LSTMConfig cfg;

    // Parameter buffers:
    // W_ih: [4H x D], b_ih: [4H]
    // W_hh: [4H x H], b_hh: [4H]
    // W_out: [V x H], b_out: [V] (linear projection to logits)
    float* d_W_ih;
    float* d_b_ih;
    float* d_W_hh;
    float* d_b_hh;
    float* d_W_out;
    float* d_b_out;

    // Gradients:
    float* d_dW_ih;
    float* d_db_ih;
    float* d_dW_hh;
    float* d_db_hh;
    float* d_dW_out;
    float* d_db_out;

    // Optimizer state (Adam):
    float* d_m_W_ih; float* d_v_W_ih;
    float* d_m_b_ih; float* d_v_b_ih;
    float* d_m_W_hh; float* d_v_W_hh;
    float* d_m_b_hh; float* d_v_b_hh;
    float* d_m_W_out; float* d_v_W_out;
    float* d_m_b_out; float* d_v_b_out;

    // Sequence States & Pre-activations across all timesteps T:
    // Shape per timestep:
    // X: [T x N x D]
    // G_ih: [T x N x 4H]
    // G_hh: [T x N x 4H]
    // G_total: [T x N x 4H]
    // G_act: [T x N x 4H] -> [i, f, g, o]
    // C: [(T+1) x N x H] (C[0] = initial c_0)
    // Tanh_C: [T x N x H]
    // H: [(T+1) x N x H] (H[0] = initial h_0)
    // Logits: [T x N x V]
    // Probs: [T x N x V]
    // Losses: [T x N]
    // Targets: [T x N]
    float* d_X_seq;
    float* d_G_ih;
    float* d_G_hh;
    float* d_G_total;
    float* d_G_act;
    float* d_C_seq;
    float* d_Tanh_C_seq;
    float* d_H_seq;
    float* d_Logits_seq;
    float* d_Probs_seq;
    float* d_Losses_seq;
    int*   d_Targets_seq;

    // Backprop scratch buffers:
    float* d_dLogits_seq;
    float* d_dH_seq;
    float* d_dC_seq;
    float* d_dG_seq;
    float* d_dX_seq;

    int opt_step;

public:
    CUDALSTM(const LSTMConfig& config);
    ~CUDALSTM();

    void init_weights(unsigned int seed = 42);
    void reset_states();
    
    // Forward sequence pass: returns mean cross-entropy loss
    float forward(const float* h_X_seq, const int* h_targets_seq, cudaStream_t stream = 0);

    // Backward sequence pass (BPTT): computes analytical gradients
    void backward(cudaStream_t stream = 0);

    // Optimizer step with gradient clipping
    void step(cudaStream_t stream = 0);

    // Single step inference
    void predict_step(
        const float* d_x_t,
        const float* d_h_prev,
        const float* d_c_prev,
        float* d_h_next,
        float* d_c_next,
        float* d_logits,
        cudaStream_t stream = 0
    );
};

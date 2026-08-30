#ifndef CNN_CUH
#define CNN_CUH

#include "../../00_common/include/cuda_utils.cuh"
#include <vector>
#include <string>

// CUDACNN: End-to-end standalone Convolutional Neural Network
// Architecture:
// Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU -> MaxPool2 -> Flatten -> FC1 -> ReLU -> FC2 -> Softmax Cross-Entropy
class CUDACNN {
public:
    int N;             // Batch size
    int C_in;          // Input channels (e.g. 1 for MNIST)
    int H_in;          // Input height (e.g. 28)
    int W_in;          // Input width (e.g. 28)
    
    // Conv1 parameters
    int C1_out;        // e.g. 8 or 16
    int K1;            // e.g. 3
    int pad1;          // e.g. 1
    int stride1;       // e.g. 1
    int H1_out, W1_out;
    int H1_pool, W1_pool;

    // Conv2 parameters
    int C2_out;        // e.g. 16 or 32
    int K2;            // e.g. 3
    int pad2;          // e.g. 1
    int stride2;       // e.g. 1
    int H2_out, W2_out;
    int H2_pool, W2_pool;

    // Fully Connected parameters
    int flat_dim;      // C2_out * H2_pool * W2_pool
    int hidden_dim;    // e.g. 64 or 128
    int num_classes;   // e.g. 10

    // Hyperparameters
    float lr;
    float momentum;
    int current_step;

    // Device memory buffers (Weights & Biases)
    float *d_W_conv1, *d_b_conv1;
    float *d_W_conv2, *d_b_conv2;
    float *d_W_fc1, *d_b_fc1;
    float *d_W_fc2, *d_b_fc2;

    // Gradients
    float *d_dW_conv1, *d_db_conv1;
    float *d_dW_conv2, *d_db_conv2;
    float *d_dW_fc1, *d_db_fc1;
    float *d_dW_fc2, *d_db_fc2;

    // Optimizer state (Momentum & Adam)
    float *d_v_W_conv1, *d_v_b_conv1;
    float *d_v_W_conv2, *d_v_b_conv2;
    float *d_v_W_fc1, *d_v_b_fc1;
    float *d_v_W_fc2, *d_v_b_fc2;

    // Activations & Intermediate Buffers
    float *d_X;
    float *d_targets;
    float *d_Z_conv1, *d_A_conv1, *d_P1;
    int   *d_mask1;
    float *d_Z_conv2, *d_A_conv2, *d_P2;
    int   *d_mask2;
    float *d_Z_fc1, *d_A_fc1;
    float *d_logits, *d_probs;
    float *d_loss;

    // Backprop Error Buffers
    float *d_dlogits;
    float *d_dZ_fc1, *d_dA_fc1;
    float *d_dP2, *d_dA_conv2, *d_dZ_conv2, *d_dX_conv2;
    float *d_dP1, *d_dA_conv1, *d_dZ_conv1;

    CUDACNN(
        int batch_size = 64,
        int in_channels = 1,
        int in_h = 28,
        int in_w = 28,
        int conv1_channels = 8,
        int conv2_channels = 16,
        int fc_hidden = 64,
        int num_classes = 10,
        float learning_rate = 0.01f,
        float momentum_val = 0.9f
    );

    ~CUDACNN();

    void init_weights(unsigned long seed = 42);
    void forward(const float* h_X);
    float backward_and_step(const float* h_targets, bool use_adam = false);
    float evaluate_accuracy(const float* h_X, const float* h_labels, int total_samples);
};

#endif // CNN_CUH

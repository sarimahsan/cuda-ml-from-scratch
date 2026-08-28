#pragma once

#include "../../00_common/include/cuda_utils.cuh"
#include <vector>
#include <string>

struct MLPConfig {
    int input_dim;       // D_in (e.g., 784 for MNIST)
    int hidden_dim;      // H (e.g., 128)
    int output_dim;      // C (e.g., 10 classes)
    int batch_size;      // N (e.g., 64, 128, or full batch)
    float learning_rate; // eta
    float momentum;      // beta (for SGD with momentum)
    int epochs;
};

class MLPCUDA {
public:
    int N;      // Batch / Dataset size
    int D_in;   // Input features
    int H;      // Hidden neurons
    int C;      // Output classes
    float lr;
    float momentum;

    // Device Memory Pointers - Forward activations & pre-activations
    float* d_X;    // [N x D_in] Input batch
    float* d_y;    // [N] Ground truth integer class indices or one-hot targets
    float* d_Z1;   // [N x H] Pre-activation Layer 1 (X * W1 + b1)
    float* d_A1;   // [N x H] Activated Layer 1 (ReLU(Z1))
    float* d_Z2;   // [N x C] Pre-activation Layer 2 (A1 * W2 + b2)
    float* d_A2;   // [N x C] Softmax Probabilities (y_hat)

    // Device Memory Pointers - Model Parameters
    float* d_W1;   // [D_in x H] Weight matrix Layer 1
    float* d_b1;   // [H] Bias vector Layer 1
    float* d_W2;   // [H x C] Weight matrix Layer 2
    float* d_b2;   // [C] Bias vector Layer 2

    // Device Memory Pointers - Gradients
    float* d_dZ2;  // [N x C] Gradient w.r.t Z2 ((y_hat - y_one_hot) / N)
    float* d_dW2;  // [H x C] Gradient w.r.t W2 (A1^T * dZ2)
    float* d_db2;  // [C] Gradient w.r.t b2 (sum over batch of dZ2)
    float* d_dA1;  // [N x H] Gradient w.r.t A1 (dZ2 * W2^T)
    float* d_dZ1;  // [N x H] Gradient w.r.t Z1 (dA1 * ReLU'(Z1))
    float* d_dW1;  // [D_in x H] Gradient w.r.t W1 (X^T * dZ1)
    float* d_db1;  // [H] Gradient w.r.t b1 (sum over batch of dZ1)

    // Device Memory Pointers - Optimizer Momentum Buffers
    float* d_v_W1; // [D_in x H] Velocity for W1
    float* d_v_b1; // [H] Velocity for b1
    float* d_v_W2; // [H x C] Velocity for W2
    float* d_v_b2; // [C] Velocity for b2

    // Device Memory Pointers - Loss reduction buffers
    float* d_loss_blocks; // Scratchpad for partial block reductions
    float* d_loss;        // [1] Final scalar loss

    // Block & Grid execution configurations
    dim3 block_dim_2d;
    dim3 grid_dim_z1;
    dim3 grid_dim_z2;

public:
    MLPCUDA(int batch_size, int input_dim, int hidden_dim, int output_dim, float learning_rate = 0.01f, float momentum = 0.9f);
    ~MLPCUDA();

    // Initialization (He / Xavier normal on GPU)
    void init_weights(unsigned int seed = 42);

    // Data Transfer to GPU
    void load_batch(const float* h_X, const float* h_y, int current_batch_size);

    // Forward Pass: X -> Z1 -> A1 (ReLU) -> Z2 -> A2 (Softmax)
    void forward(int current_batch_size, cudaStream_t stream = 0);

    // Compute Categorical Cross-Entropy Loss
    float compute_loss(int current_batch_size, cudaStream_t stream = 0);

    // Backward Pass: computes analytical gradients for all layers via chain rule
    void backward(int current_batch_size, cudaStream_t stream = 0);

    // Optimizer step: updates weights and biases using SGD with Momentum
    void step(cudaStream_t stream = 0);

    // Inference & Evaluation: outputs class predictions (argmax)
    void predict(const float* h_X, int* h_preds, int num_samples);

    // Compute classification accuracy on host
    float evaluate(const float* h_X, const float* h_y, int num_samples);
};

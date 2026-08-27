#pragma once

#include "../../00_common/include/cuda_utils.cuh"
#include <vector>

struct LogisticRegressionConfig {
    int num_samples;      // N (batch or dataset size)
    int num_features;     // D (dimension)
    float learning_rate;  // eta
    int epochs;
    int batch_size;       // Full batch or mini-batch
    int block_size;       // Threads per block (e.g. 256)
};

class LogisticRegressionCUDA {
public:
    int N; // Number of samples
    int D; // Number of features
    float lr;

    // Device pointers
    float* d_X;           // [N x D] Features
    float* d_y;           // [N] Ground truth labels {0, 1}
    float* d_w;           // [D] Model weights
    float* d_b;           // [1] Model bias
    float* d_y_hat;       // [N] Predicted probabilities sigma(Xw + b)
    float* d_grad_w;      // [D] Gradients w.r.t weights
    float* d_grad_b;      // [1] Gradient w.r.t bias
    float* d_loss;        // [1] Scalar loss buffer (accumulated via reduction)
    float* d_loss_blocks; // Scratchpad for partial block reductions

    int block_size;
    int grid_size_samples;

public:
    LogisticRegressionCUDA(int num_samples, int num_features, float learning_rate = 0.1f, int threads_per_block = 256);
    ~LogisticRegressionCUDA();

    // Data transfer
    void load_data(const float* h_X, const float* h_y);

    // Forward pass: y_hat = sigmoid(X * w + b)
    void forward(cudaStream_t stream = 0);

    // Compute Binary Cross-Entropy Loss
    float compute_loss(cudaStream_t stream = 0);

    // Backward pass: computes gradients d_grad_w and d_grad_b
    void backward(cudaStream_t stream = 0);

    // Optimizer step: updates d_w and d_b using SGD
    void step(cudaStream_t stream = 0);

    // Predict & fetch predictions back to host
    void predict(const float* h_X_test, float* h_preds, int N_test);

    // Get current weights & bias to host
    void get_weights(std::vector<float>& h_w, float& h_b);
};

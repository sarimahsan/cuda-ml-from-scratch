#include "../include/mlp.cuh"
#include "../include/linear.cuh"
#include "../include/activations.cuh"
#include "../include/softmax_loss.cuh"
#include "../include/optimizers.cuh"
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <algorithm>

// -------------------------------------------------------------------------
// Weight Initialization Kernels (He Normal & Zeros)
// -------------------------------------------------------------------------
__global__ void init_he_weights_kernel(float* W, int size, float stddev, unsigned long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        W[idx] = curand_normal(&state) * stddev;
    }
}

__global__ void init_zeros_kernel(float* b, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        b[idx] = 0.0f;
    }
}

// -------------------------------------------------------------------------
// MLPCUDA Implementation
// -------------------------------------------------------------------------
MLPCUDA::MLPCUDA(int batch_size, int input_dim, int hidden_dim, int output_dim, float learning_rate, float momentum_val)
    : N(batch_size), D_in(input_dim), H(hidden_dim), C(output_dim), lr(learning_rate), momentum(momentum_val)
{
    // Allocate device memory for forward activations
    d_X  = allocate_device_memory<float>(N * D_in);
    d_y  = allocate_device_memory<float>(N);
    d_Z1 = allocate_device_memory<float>(N * H);
    d_A1 = allocate_device_memory<float>(N * H);
    d_Z2 = allocate_device_memory<float>(N * C);
    d_A2 = allocate_device_memory<float>(N * C);

    // Allocate device memory for weights & biases
    d_W1 = allocate_device_memory<float>(D_in * H);
    d_b1 = allocate_device_memory<float>(H);
    d_W2 = allocate_device_memory<float>(H * C);
    d_b2 = allocate_device_memory<float>(C);

    // Allocate device memory for gradients
    d_dZ2 = allocate_device_memory<float>(N * C);
    d_dW2 = allocate_device_memory<float>(H * C);
    d_db2 = allocate_device_memory<float>(C);
    d_dA1 = allocate_device_memory<float>(N * H);
    d_dZ1 = allocate_device_memory<float>(N * H);
    d_dW1 = allocate_device_memory<float>(D_in * H);
    d_db1 = allocate_device_memory<float>(H);

    // Allocate device memory for optimizer momentum buffers
    d_v_W1 = allocate_device_memory<float>(D_in * H);
    d_v_b1 = allocate_device_memory<float>(H);
    d_v_W2 = allocate_device_memory<float>(H * C);
    d_v_b2 = allocate_device_memory<float>(C);

    // Allocate loss scratchpad
    int num_loss_blocks = (N + 255) / 256;
    d_loss_blocks = allocate_device_memory<float>(num_loss_blocks);
    d_loss = allocate_device_memory<float>(1);

    // Zero-out momentum buffers
    CUDA_CHECK(cudaMemset(d_v_W1, 0, D_in * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b1, 0, H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W2, 0, H * C * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b2, 0, C * sizeof(float)));

    // Initialize model parameters
    init_weights();
}

MLPCUDA::~MLPCUDA() {
    free_device_memory(d_X);
    free_device_memory(d_y);
    free_device_memory(d_Z1);
    free_device_memory(d_A1);
    free_device_memory(d_Z2);
    free_device_memory(d_A2);

    free_device_memory(d_W1);
    free_device_memory(d_b1);
    free_device_memory(d_W2);
    free_device_memory(d_b2);

    free_device_memory(d_dZ2);
    free_device_memory(d_dW2);
    free_device_memory(d_db2);
    free_device_memory(d_dA1);
    free_device_memory(d_dZ1);
    free_device_memory(d_dW1);
    free_device_memory(d_db1);

    free_device_memory(d_v_W1);
    free_device_memory(d_v_b1);
    free_device_memory(d_v_W2);
    free_device_memory(d_v_b2);

    free_device_memory(d_loss_blocks);
    free_device_memory(d_loss);
}

void MLPCUDA::init_weights(unsigned int seed) {
    int threads = 256;

    // He normal initialization: std = sqrt(2 / fan_in)
    float std1 = std::sqrt(2.0f / static_cast<float>(D_in));
    int blocks_W1 = (D_in * H + threads - 1) / threads;
    init_he_weights_kernel<<<blocks_W1, threads>>>(d_W1, D_in * H, std1, seed);

    float std2 = std::sqrt(2.0f / static_cast<float>(H));
    int blocks_W2 = (H * C + threads - 1) / threads;
    init_he_weights_kernel<<<blocks_W2, threads>>>(d_W2, H * C, std2, seed + 100);

    // Biases initialized to zero
    int blocks_b1 = (H + threads - 1) / threads;
    init_zeros_kernel<<<blocks_b1, threads>>>(d_b1, H);

    int blocks_b2 = (C + threads - 1) / threads;
    init_zeros_kernel<<<blocks_b2, threads>>>(d_b2, C);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void MLPCUDA::load_batch(const float* h_X, const float* h_y, int current_batch_size) {
    CUDA_CHECK(cudaMemcpy(d_X, h_X, current_batch_size * D_in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, current_batch_size * sizeof(float), cudaMemcpyHostToDevice));
}

void MLPCUDA::forward(int current_batch_size, cudaStream_t stream) {
    // 1. Layer 1 Linear Forward: Z1 = X * W1 + b1
    launch_linear_forward(d_X, d_W1, d_b1, d_Z1, current_batch_size, D_in, H, stream);

    // 2. Layer 1 Activation: A1 = ReLU(Z1)
    launch_activation_forward(d_Z1, d_A1, current_batch_size * H, ActivationType::RELU, stream);

    // 3. Layer 2 Linear Forward: Z2 = A1 * W2 + b2
    launch_linear_forward(d_A1, d_W2, d_b2, d_Z2, current_batch_size, H, C, stream);

    // 4. Softmax Output: A2 = Softmax(Z2)
    launch_softmax_forward(d_Z2, d_A2, current_batch_size, C, stream);
}

float MLPCUDA::compute_loss(int current_batch_size, cudaStream_t stream) {
    return launch_cross_entropy_loss(d_A2, d_y, d_loss_blocks, d_loss, current_batch_size, C, stream);
}

void MLPCUDA::backward(int current_batch_size, cudaStream_t stream) {
    // 1. Output Gradient: dZ2 = (A2 - y_onehot) / N
    launch_softmax_cross_entropy_backward(d_A2, d_y, d_dZ2, current_batch_size, C, stream);

    // 2. Layer 2 Linear Backward: dW2 = A1^T * dZ2, db2 = sum(dZ2), dA1 = dZ2 * W2^T
    launch_linear_backward(d_dZ2, d_A1, d_W2, d_dW2, d_db2, d_dA1, current_batch_size, H, C, true, stream);

    // 3. Layer 1 Activation Backward: dZ1 = dA1 * ReLU'(Z1)
    launch_activation_backward(d_dA1, d_Z1, d_dZ1, current_batch_size * H, ActivationType::RELU, stream);

    // 4. Layer 1 Linear Backward: dW1 = X^T * dZ1, db1 = sum(dZ1)
    launch_linear_backward(d_dZ1, d_X, d_W1, d_dW1, d_db1, nullptr, current_batch_size, D_in, H, false, stream);
}

void MLPCUDA::step(cudaStream_t stream) {
    // Update Layer 1 parameters
    launch_sgd_momentum(d_W1, d_v_W1, d_dW1, lr, momentum, D_in * H, stream);
    launch_sgd_momentum(d_b1, d_v_b1, d_db1, lr, momentum, H, stream);

    // Update Layer 2 parameters
    launch_sgd_momentum(d_W2, d_v_W2, d_dW2, lr, momentum, H * C, stream);
    launch_sgd_momentum(d_b2, d_v_b2, d_db2, lr, momentum, C, stream);
}

void MLPCUDA::predict(const float* h_X, int* h_preds, int num_samples) {
    int* d_preds = allocate_device_memory<int>(num_samples);

    for (int offset = 0; offset < num_samples; offset += N) {
        int cur_n = std::min(N, num_samples - offset);
        CUDA_CHECK(cudaMemcpy(d_X, h_X + offset * D_in, cur_n * D_in * sizeof(float), cudaMemcpyHostToDevice));

        forward(cur_n);
        launch_argmax(d_A2, d_preds + offset, cur_n, C);
    }

    CUDA_CHECK(cudaMemcpy(h_preds, d_preds, num_samples * sizeof(int), cudaMemcpyDeviceToHost));
    free_device_memory(d_preds);
}

float MLPCUDA::evaluate(const float* h_X, const float* h_y, int num_samples) {
    std::vector<int> preds(num_samples);
    predict(h_X, preds.data(), num_samples);

    int correct = 0;
    for (int i = 0; i < num_samples; ++i) {
        if (preds[i] == static_cast<int>(h_y[i])) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(num_samples) * 100.0f;
}

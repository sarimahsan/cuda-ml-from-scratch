#include "../include/cnn.cuh"
#include "../include/conv2d.cuh"
#include "../include/pool.cuh"
#include "../include/linear.cuh"
#include "../include/activations.cuh"
#include "../include/softmax_loss.cuh"
#include "../include/optimizers.cuh"
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <algorithm>

// -------------------------------------------------------------------------
// Weight Initialization Kernels (Kaiming / He Normal & Zeros)
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
// CUDACNN Constructor & Destructor
// -------------------------------------------------------------------------
CUDACNN::CUDACNN(
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int conv1_channels,
    int conv2_channels,
    int fc_hidden,
    int num_classes_val,
    float learning_rate,
    float momentum_val
) : N(batch_size), C_in(in_channels), H_in(in_h), W_in(in_w),
    C1_out(conv1_channels), K1(3), pad1(1), stride1(1),
    C2_out(conv2_channels), K2(3), pad2(1), stride2(1),
    hidden_dim(fc_hidden), num_classes(num_classes_val),
    lr(learning_rate), momentum(momentum_val), current_step(0)
{
    // Conv1 spatial output: (H_in + 2*pad1 - K1)/stride1 + 1 = (28 + 2*1 - 3)/1 + 1 = 28
    H1_out = (H_in + 2 * pad1 - K1) / stride1 + 1;
    W1_out = (W_in + 2 * pad1 - K1) / stride1 + 1;

    // MaxPool1 (2x2, stride 2): 28 / 2 = 14
    H1_pool = H1_out / 2;
    W1_pool = W1_out / 2;

    // Conv2 spatial output: (14 + 2*1 - 3)/1 + 1 = 14
    H2_out = (H1_pool + 2 * pad2 - K2) / stride2 + 1;
    W2_out = (W1_pool + 2 * pad2 - K2) / stride2 + 1;

    // MaxPool2 (2x2, stride 2): 14 / 2 = 7
    H2_pool = H2_out / 2;
    W2_pool = W2_out / 2;

    // Flattened dimension: C2_out * H2_pool * W2_pool (e.g. 16 * 7 * 7 = 784)
    flat_dim = C2_out * H2_pool * W2_pool;

    // Allocate Device Memory for Input and Targets
    d_X = allocate_device_memory<float>(N * C_in * H_in * W_in);
    d_targets = allocate_device_memory<float>(N * num_classes);

    // Allocate Conv1 buffers
    d_W_conv1 = allocate_device_memory<float>(C1_out * C_in * K1 * K1);
    d_b_conv1 = allocate_device_memory<float>(C1_out);
    d_dW_conv1 = allocate_device_memory<float>(C1_out * C_in * K1 * K1);
    d_db_conv1 = allocate_device_memory<float>(C1_out);
    d_v_W_conv1 = allocate_device_memory<float>(C1_out * C_in * K1 * K1);
    d_v_b_conv1 = allocate_device_memory<float>(C1_out);

    d_Z_conv1 = allocate_device_memory<float>(N * C1_out * H1_out * W1_out);
    d_A_conv1 = allocate_device_memory<float>(N * C1_out * H1_out * W1_out);
    d_P1      = allocate_device_memory<float>(N * C1_out * H1_pool * W1_pool);
    d_mask1   = allocate_device_memory<int>(N * C1_out * H1_pool * W1_pool);

    d_dP1      = allocate_device_memory<float>(N * C1_out * H1_pool * W1_pool);
    d_dA_conv1 = allocate_device_memory<float>(N * C1_out * H1_out * W1_out);
    d_dZ_conv1 = allocate_device_memory<float>(N * C1_out * H1_out * W1_out);

    // Allocate Conv2 buffers
    d_W_conv2 = allocate_device_memory<float>(C2_out * C1_out * K2 * K2);
    d_b_conv2 = allocate_device_memory<float>(C2_out);
    d_dW_conv2 = allocate_device_memory<float>(C2_out * C1_out * K2 * K2);
    d_db_conv2 = allocate_device_memory<float>(C2_out);
    d_v_W_conv2 = allocate_device_memory<float>(C2_out * C1_out * K2 * K2);
    d_v_b_conv2 = allocate_device_memory<float>(C2_out);

    d_Z_conv2 = allocate_device_memory<float>(N * C2_out * H2_out * W2_out);
    d_A_conv2 = allocate_device_memory<float>(N * C2_out * H2_out * W2_out);
    d_P2      = allocate_device_memory<float>(N * flat_dim);
    d_mask2   = allocate_device_memory<int>(N * flat_dim);

    d_dP2      = allocate_device_memory<float>(N * flat_dim);
    d_dA_conv2 = allocate_device_memory<float>(N * C2_out * H2_out * W2_out);
    d_dZ_conv2 = allocate_device_memory<float>(N * C2_out * H2_out * W2_out);
    d_dX_conv2 = allocate_device_memory<float>(N * C1_out * H1_pool * W1_pool);

    // Allocate FC1 & FC2 buffers
    d_W_fc1 = allocate_device_memory<float>(flat_dim * hidden_dim);
    d_b_fc1 = allocate_device_memory<float>(hidden_dim);
    d_dW_fc1 = allocate_device_memory<float>(flat_dim * hidden_dim);
    d_db_fc1 = allocate_device_memory<float>(hidden_dim);
    d_v_W_fc1 = allocate_device_memory<float>(flat_dim * hidden_dim);
    d_v_b_fc1 = allocate_device_memory<float>(hidden_dim);

    d_Z_fc1 = allocate_device_memory<float>(N * hidden_dim);
    d_A_fc1 = allocate_device_memory<float>(N * hidden_dim);
    d_dZ_fc1 = allocate_device_memory<float>(N * hidden_dim);
    d_dA_fc1 = allocate_device_memory<float>(N * hidden_dim);

    d_W_fc2 = allocate_device_memory<float>(hidden_dim * num_classes);
    d_b_fc2 = allocate_device_memory<float>(num_classes);
    d_dW_fc2 = allocate_device_memory<float>(hidden_dim * num_classes);
    d_db_fc2 = allocate_device_memory<float>(num_classes);
    d_v_W_fc2 = allocate_device_memory<float>(hidden_dim * num_classes);
    d_v_b_fc2 = allocate_device_memory<float>(num_classes);

    d_logits = allocate_device_memory<float>(N * num_classes);
    d_probs  = allocate_device_memory<float>(N * num_classes);
    d_dlogits = allocate_device_memory<float>(N * num_classes);
    d_loss   = allocate_device_memory<float>(1);

    // Zero-initialize optimizer buffers
    CUDA_CHECK(cudaMemset(d_v_W_conv1, 0, C1_out * C_in * K1 * K1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_conv1, 0, C1_out * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_conv2, 0, C2_out * C1_out * K2 * K2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_conv2, 0, C2_out * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_fc1, 0, flat_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_fc1, 0, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W_fc2, 0, hidden_dim * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b_fc2, 0, num_classes * sizeof(float)));
}

CUDACNN::~CUDACNN() {
    free_device_memory(d_X);
    free_device_memory(d_targets);

    free_device_memory(d_W_conv1); free_device_memory(d_b_conv1);
    free_device_memory(d_dW_conv1); free_device_memory(d_db_conv1);
    free_device_memory(d_v_W_conv1); free_device_memory(d_v_b_conv1);
    free_device_memory(d_Z_conv1); free_device_memory(d_A_conv1);
    free_device_memory(d_P1); free_device_memory(d_mask1);
    free_device_memory(d_dP1); free_device_memory(d_dA_conv1); free_device_memory(d_dZ_conv1);

    free_device_memory(d_W_conv2); free_device_memory(d_b_conv2);
    free_device_memory(d_dW_conv2); free_device_memory(d_db_conv2);
    free_device_memory(d_v_W_conv2); free_device_memory(d_v_b_conv2);
    free_device_memory(d_Z_conv2); free_device_memory(d_A_conv2);
    free_device_memory(d_P2); free_device_memory(d_mask2);
    free_device_memory(d_dP2); free_device_memory(d_dA_conv2); free_device_memory(d_dZ_conv2); free_device_memory(d_dX_conv2);

    free_device_memory(d_W_fc1); free_device_memory(d_b_fc1);
    free_device_memory(d_dW_fc1); free_device_memory(d_db_fc1);
    free_device_memory(d_v_W_fc1); free_device_memory(d_v_b_fc1);
    free_device_memory(d_Z_fc1); free_device_memory(d_A_fc1);
    free_device_memory(d_dZ_fc1); free_device_memory(d_dA_fc1);

    free_device_memory(d_W_fc2); free_device_memory(d_b_fc2);
    free_device_memory(d_dW_fc2); free_device_memory(d_db_fc2);
    free_device_memory(d_v_W_fc2); free_device_memory(d_v_b_fc2);

    free_device_memory(d_logits); free_device_memory(d_probs);
    free_device_memory(d_dlogits); free_device_memory(d_loss);
}

void CUDACNN::init_weights(unsigned long seed) {
    int block = 256;

    // Conv1 He init: fan_in = C_in * K1 * K1
    float std1 = std::sqrt(2.0f / (float)(C_in * K1 * K1));
    int s_conv1 = C1_out * C_in * K1 * K1;
    init_he_weights_kernel<<<(s_conv1 + block - 1) / block, block>>>(d_W_conv1, s_conv1, std1, seed);
    init_zeros_kernel<<<(C1_out + block - 1) / block, block>>>(d_b_conv1, C1_out);

    // Conv2 He init: fan_in = C1_out * K2 * K2
    float std2 = std::sqrt(2.0f / (float)(C1_out * K2 * K2));
    int s_conv2 = C2_out * C1_out * K2 * K2;
    init_he_weights_kernel<<<(s_conv2 + block - 1) / block, block>>>(d_W_conv2, s_conv2, std2, seed + 1);
    init_zeros_kernel<<<(C2_out + block - 1) / block, block>>>(d_b_conv2, C2_out);

    // FC1 He init: fan_in = flat_dim
    float std_fc1 = std::sqrt(2.0f / (float)flat_dim);
    int s_fc1 = flat_dim * hidden_dim;
    init_he_weights_kernel<<<(s_fc1 + block - 1) / block, block>>>(d_W_fc1, s_fc1, std_fc1, seed + 2);
    init_zeros_kernel<<<(hidden_dim + block - 1) / block, block>>>(d_b_fc1, hidden_dim);

    // FC2 He init: fan_in = hidden_dim
    float std_fc2 = std::sqrt(2.0f / (float)hidden_dim);
    int s_fc2 = hidden_dim * num_classes;
    init_he_weights_kernel<<<(s_fc2 + block - 1) / block, block>>>(d_W_fc2, s_fc2, std_fc2, seed + 3);
    init_zeros_kernel<<<(num_classes + block - 1) / block, block>>>(d_b_fc2, num_classes);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDACNN::forward(const float* h_X) {
    if (h_X != nullptr) {
        copy_to_device(d_X, h_X, N * C_in * H_in * W_in);
    }

    // 1. Conv1: [N, C_in, 28, 28] -> [N, C1_out, 28, 28]
    conv2d_forward(d_X, d_W_conv1, d_b_conv1, d_Z_conv1, N, C_in, H_in, W_in, C1_out, K1, K1, stride1, pad1, H1_out, W1_out);
    relu_forward(d_Z_conv1, d_A_conv1, N * C1_out * H1_out * W1_out);

    // 2. MaxPool1: [N, C1_out, 28, 28] -> [N, C1_out, 14, 14]
    maxpool2d_forward(d_A_conv1, d_P1, d_mask1, N, C1_out, H1_out, W1_out, 2, 2, 2, 0, H1_pool, W1_pool);

    // 3. Conv2: [N, C1_out, 14, 14] -> [N, C2_out, 14, 14]
    conv2d_forward(d_P1, d_W_conv2, d_b_conv2, d_Z_conv2, N, C1_out, H1_pool, W1_pool, C2_out, K2, K2, stride2, pad2, H2_out, W2_out);
    relu_forward(d_Z_conv2, d_A_conv2, N * C2_out * H2_out * W2_out);

    // 4. MaxPool2: [N, C2_out, 14, 14] -> [N, C2_out, 7, 7] (flattened as d_P2: [N, flat_dim])
    maxpool2d_forward(d_A_conv2, d_P2, d_mask2, N, C2_out, H2_out, W2_out, 2, 2, 2, 0, H2_pool, W2_pool);

    // 5. FC1: [N, flat_dim] -> [N, hidden_dim]
    linear_forward(d_P2, d_W_fc1, d_b_fc1, d_Z_fc1, N, flat_dim, hidden_dim);
    relu_forward(d_Z_fc1, d_A_fc1, N * hidden_dim);

    // 6. FC2: [N, hidden_dim] -> [N, num_classes] (logits)
    linear_forward(d_A_fc1, d_W_fc2, d_b_fc2, d_logits, N, hidden_dim, num_classes);
    softmax_forward(d_logits, d_probs, N, num_classes);
}

float CUDACNN::backward_and_step(const float* h_targets, bool use_adam) {
    current_step++;
    if (h_targets != nullptr) {
        copy_to_device(d_targets, h_targets, N * num_classes);
    }

    // 1. Loss & Softmax Cross Entropy Gradient: d_dlogits = (probs - targets) / N
    softmax_cross_entropy_loss_and_grad(d_logits, d_targets, d_probs, d_dlogits, d_loss, N, num_classes);

    // 2. FC2 Backward: d_dlogits -> d_dW_fc2, d_db_fc2, d_dA_fc1
    linear_backward(d_dlogits, d_A_fc1, d_W_fc2, d_dW_fc2, d_db_fc2, d_dA_fc1, N, hidden_dim, num_classes, true);

    // 3. FC1 Activation Backward
    relu_backward(d_dA_fc1, d_Z_fc1, d_dZ_fc1, N * hidden_dim);

    // 4. FC1 Backward: d_dZ_fc1 -> d_dW_fc1, d_db_fc1, d_dP2
    linear_backward(d_dZ_fc1, d_P2, d_W_fc1, d_dW_fc1, d_db_fc1, d_dP2, N, flat_dim, hidden_dim, true);

    // 5. MaxPool2 Backward: d_dP2 -> d_dA_conv2
    maxpool2d_backward(d_dP2, d_mask2, d_dA_conv2, N, C2_out, H2_out, W2_out, H2_pool, W2_pool);

    // 6. Conv2 Activation Backward
    relu_backward(d_dA_conv2, d_Z_conv2, d_dZ_conv2, N * C2_out * H2_out * W2_out);

    // 7. Conv2 Backward: d_dZ_conv2 -> d_dW_conv2, d_db_conv2, d_dX_conv2 (which is d_dP1)
    conv2d_backward_filter_bias(d_P1, d_dZ_conv2, d_dW_conv2, d_db_conv2, N, C1_out, H1_pool, W1_pool, C2_out, K2, K2, stride2, pad2, H2_out, W2_out);
    conv2d_backward_data(d_dZ_conv2, d_W_conv2, d_dX_conv2, N, C1_out, H1_pool, W1_pool, C2_out, K2, K2, stride2, pad2, H2_out, W2_out);

    // 8. MaxPool1 Backward: d_dX_conv2 (d_dP1) -> d_dA_conv1
    maxpool2d_backward(d_dX_conv2, d_mask1, d_dA_conv1, N, C1_out, H1_out, W1_out, H1_pool, W1_pool);

    // 9. Conv1 Activation Backward
    relu_backward(d_dA_conv1, d_Z_conv1, d_dZ_conv1, N * C1_out * H1_out * W1_out);

    // 10. Conv1 Backward Filter & Bias: d_dZ_conv1 -> d_dW_conv1, d_db_conv1
    conv2d_backward_filter_bias(d_X, d_dZ_conv1, d_dW_conv1, d_db_conv1, N, C_in, H_in, W_in, C1_out, K1, K1, stride1, pad1, H1_out, W1_out);

    // 11. Optimizer Updates (SGD Momentum)
    sgd_momentum_update(d_W_conv1, d_v_W_conv1, d_dW_conv1, C1_out * C_in * K1 * K1, lr, momentum);
    sgd_momentum_update(d_b_conv1, d_v_b_conv1, d_db_conv1, C1_out, lr, momentum);

    sgd_momentum_update(d_W_conv2, d_v_W_conv2, d_dW_conv2, C2_out * C1_out * K2 * K2, lr, momentum);
    sgd_momentum_update(d_b_conv2, d_v_b_conv2, d_db_conv2, C2_out, lr, momentum);

    sgd_momentum_update(d_W_fc1, d_v_W_fc1, d_dW_fc1, flat_dim * hidden_dim, lr, momentum);
    sgd_momentum_update(d_b_fc1, d_v_b_fc1, d_db_fc1, hidden_dim, lr, momentum);

    sgd_momentum_update(d_W_fc2, d_v_W_fc2, d_dW_fc2, hidden_dim * num_classes, lr, momentum);
    sgd_momentum_update(d_b_fc2, d_v_b_fc2, d_db_fc2, num_classes, lr, momentum);

    float h_loss = 0.0f;
    copy_to_host(&h_loss, d_loss, 1);
    return h_loss;
}

float CUDACNN::evaluate_accuracy(const float* h_X, const float* h_labels, int total_samples) {
    int correct = 0;
    int num_batches = total_samples / N;

    std::vector<float> h_probs(N * num_classes);

    for (int b = 0; b < num_batches; ++b) {
        const float* batch_X = h_X + b * N * C_in * H_in * W_in;
        const float* batch_labels = h_labels + b * N;

        forward(batch_X);
        copy_to_host(h_probs.data(), d_probs, N * num_classes);

        for (int i = 0; i < N; ++i) {
            int pred = 0;
            float max_p = h_probs[i * num_classes];
            for (int c = 1; c < num_classes; ++c) {
                if (h_probs[i * num_classes + c] > max_p) {
                    max_p = h_probs[i * num_classes + c];
                    pred = c;
                }
            }
            if (pred == (int)batch_labels[i]) {
                correct++;
            }
        }
    }

    return (float)correct / (float)(num_batches * N);
}

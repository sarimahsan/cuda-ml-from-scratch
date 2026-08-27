#include "../include/logistic_regression.cuh"
#include <cmath>
#include <iostream>

#define WARP_SIZE 32
#define EPSILON 1e-7f

// -------------------------------------------------------------------------
// Device Helper Functions: Warp-Level Reductions
// -------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE]; // Shared memory for warp sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction by the first warp
    float sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

// -------------------------------------------------------------------------
// 1. Forward Kernel: Computes z = X * w + b and y_hat = sigmoid(z)
// -------------------------------------------------------------------------
__global__ void forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y_hat,
    int N,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float z = *b; // broadcast scalar bias
        #pragma unroll 4
        for (int j = 0; j < D; ++j) {
            z += X[idx * D + j] * w[j];
        }
        // Numerically stable Sigmoid: 1 / (1 + exp(-z))
        y_hat[idx] = 1.0f / (1.0f + __expf(-z));
    }
}

// -------------------------------------------------------------------------
// 2. Loss Kernel: Computes BCE Loss with Block-Level Reduction
// L = - (1/N) * sum( y * log(y_hat + eps) + (1 - y) * log(1 - y_hat + eps) )
// -------------------------------------------------------------------------
__global__ void bce_loss_kernel(
    const float* __restrict__ y_hat,
    const float* __restrict__ y,
    float* __restrict__ block_losses,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;

    if (idx < N) {
        float p = y_hat[idx];
        float target = y[idx];
        // Clip probabilities to avoid log(0)
        p = fminf(fmaxf(p, EPSILON), 1.0f - EPSILON);
        local_loss = -(target * __logf(p) + (1.0f - target) * __logf(1.0f - p));
    }

    float block_sum = block_reduce_sum(local_loss);

    if (threadIdx.x == 0) {
        block_losses[blockIdx.x] = block_sum;
    }
}

// Final reduction across blocks
__global__ void final_loss_reduction_kernel(
    const float* __restrict__ block_losses,
    float* __restrict__ total_loss,
    int num_blocks,
    int N
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_losses[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *total_loss = sum / static_cast<float>(N);
    }
}

// -------------------------------------------------------------------------
// 3. Backward Kernel: Computes Gradients for Weights and Bias
// grad_w[j] = (1/N) * sum_i( (y_hat_i - y_i) * X_i,j )
// grad_b    = (1/N) * sum_i( y_hat_i - y_i )
//
// Grid: (D + 1) blocks, where blockIdx.x < D computes grad_w[j], and blockIdx.x == D computes grad_b
// -------------------------------------------------------------------------
__global__ void backward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y_hat,
    const float* __restrict__ y,
    float* __restrict__ grad_w,
    float* __restrict__ grad_b,
    int N,
    int D
) {
    int feature_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float thread_sum = 0.0f;

    if (feature_idx < D) {
        // Gradient with respect to weight w_j
        for (int i = tid; i < N; i += stride) {
            float err = y_hat[i] - y[i];
            thread_sum += err * X[i * D + feature_idx];
        }
        float block_sum = block_reduce_sum(thread_sum);
        if (tid == 0) {
            grad_w[feature_idx] = block_sum / static_cast<float>(N);
        }
    } else if (feature_idx == D) {
        // Gradient with respect to bias b
        for (int i = tid; i < N; i += stride) {
            thread_sum += (y_hat[i] - y[i]);
        }
        float block_sum = block_reduce_sum(thread_sum);
        if (tid == 0) {
            *grad_b = block_sum / static_cast<float>(N);
        }
    }
}

// -------------------------------------------------------------------------
// 4. Optimizer Kernel: SGD parameter update
// w[j] = w[j] - lr * grad_w[j]
// b    = b    - lr * grad_b
// -------------------------------------------------------------------------
__global__ void sgd_update_kernel(
    float* __restrict__ w,
    float* __restrict__ b,
    const float* __restrict__ grad_w,
    const float* __restrict__ grad_b,
    float lr,
    int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        w[idx] -= lr * grad_w[idx];
    }
    if (idx == 0) {
        *b -= lr * (*grad_b);
    }
}

// =========================================================================
// Host Implementation for LogisticRegressionCUDA
// =========================================================================

LogisticRegressionCUDA::LogisticRegressionCUDA(int num_samples, int num_features, float learning_rate, int threads_per_block)
    : N(num_samples), D(num_features), lr(learning_rate), block_size(threads_per_block) {
    
    grid_size_samples = (N + block_size - 1) / block_size;

    // Allocate GPU buffers
    d_X           = allocate_device_memory<float>(N * D);
    d_y           = allocate_device_memory<float>(N);
    d_w           = allocate_device_memory<float>(D);
    d_b           = allocate_device_memory<float>(1);
    d_y_hat       = allocate_device_memory<float>(N);
    d_grad_w      = allocate_device_memory<float>(D);
    d_grad_b      = allocate_device_memory<float>(1);
    d_loss        = allocate_device_memory<float>(1);
    d_loss_blocks = allocate_device_memory<float>(grid_size_samples);

    // Initialize weights and bias to zeros on GPU
    CUDA_CHECK(cudaMemset(d_w, 0, D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, 1 * sizeof(float)));
}

LogisticRegressionCUDA::~LogisticRegressionCUDA() {
    free_device_memory(d_X);
    free_device_memory(d_y);
    free_device_memory(d_w);
    free_device_memory(d_b);
    free_device_memory(d_y_hat);
    free_device_memory(d_grad_w);
    free_device_memory(d_grad_b);
    free_device_memory(d_loss);
    free_device_memory(d_loss_blocks);
}

void LogisticRegressionCUDA::load_data(const float* h_X, const float* h_y) {
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
}

void LogisticRegressionCUDA::forward(cudaStream_t stream) {
    forward_kernel<<<grid_size_samples, block_size, 0, stream>>>(
        d_X, d_w, d_b, d_y_hat, N, D
    );
    CUDA_KERNEL_CHECK();
}

float LogisticRegressionCUDA::compute_loss(cudaStream_t stream) {
    bce_loss_kernel<<<grid_size_samples, block_size, 0, stream>>>(
        d_y_hat, d_y, d_loss_blocks, N
    );
    CUDA_KERNEL_CHECK();

    final_loss_reduction_kernel<<<1, 256, 0, stream>>>(
        d_loss_blocks, d_loss, grid_size_samples, N
    );
    CUDA_KERNEL_CHECK();

    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_loss;
}

void LogisticRegressionCUDA::backward(cudaStream_t stream) {
    // Launch (D + 1) blocks with 256 threads each to parallelize feature gradient reductions
    backward_kernel<<<D + 1, 256, 0, stream>>>(
        d_X, d_y_hat, d_y, d_grad_w, d_grad_b, N, D
    );
    CUDA_KERNEL_CHECK();
}

void LogisticRegressionCUDA::step(cudaStream_t stream) {
    int update_blocks = (D + block_size - 1) / block_size;
    if (update_blocks == 0) update_blocks = 1;
    sgd_update_kernel<<<update_blocks, block_size, 0, stream>>>(
        d_w, d_b, d_grad_w, d_grad_b, lr, D
    );
    CUDA_KERNEL_CHECK();
}

void LogisticRegressionCUDA::predict(const float* h_X_test, float* h_preds, int N_test) {
    float* d_X_test = allocate_device_memory<float>(N_test * D);
    float* d_preds  = allocate_device_memory<float>(N_test);

    CUDA_CHECK(cudaMemcpy(d_X_test, h_X_test, N_test * D * sizeof(float), cudaMemcpyHostToDevice));

    int test_grid = (N_test + block_size - 1) / block_size;
    forward_kernel<<<test_grid, block_size>>>(
        d_X_test, d_w, d_b, d_preds, N_test, D
    );
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_preds, d_preds, N_test * sizeof(float), cudaMemcpyDeviceToHost));

    free_device_memory(d_X_test);
    free_device_memory(d_preds);
}

void LogisticRegressionCUDA::get_weights(std::vector<float>& h_w, float& h_b) {
    h_w.resize(D);
    CUDA_CHECK(cudaMemcpy(h_w.data(), d_w, D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost));
}

#include "../include/softmax_loss.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include <cmath>

#define WARP_SIZE 32
#define EPSILON 1e-7f

// -------------------------------------------------------------------------
// Reduction Helpers
// -------------------------------------------------------------------------
__inline__ __device__ float softmax_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float softmax_block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = softmax_warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        sum = softmax_warp_reduce_sum(sum);
    }
    return sum;
}

// -------------------------------------------------------------------------
// 1. Numerically Stable Softmax Forward Kernel
// -------------------------------------------------------------------------
__global__ void softmax_forward_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int N, int C
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        // Step 1: Find row maximum to prevent overflow
        float max_val = -1e30f;
        for (int c = 0; c < C; ++c) {
            float v = logits[row * C + c];
            if (v > max_val) max_val = v;
        }

        // Step 2: Compute exponentials and normalizer
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float e = __expf(logits[row * C + c] - max_val);
            probs[row * C + c] = e;
            sum_exp += e;
        }

        // Step 3: Normalize to probability distribution
        float inv_sum = 1.0f / (sum_exp + EPSILON);
        for (int c = 0; c < C; ++c) {
            probs[row * C + c] *= inv_sum;
        }
    }
}

// -------------------------------------------------------------------------
// 2. Categorical Cross-Entropy Loss Kernel with Reductions
// -------------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ targets,
    float* __restrict__ block_losses,
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;

    if (idx < N) {
        int label = static_cast<int>(targets[idx]);
        if (label >= 0 && label < C) {
            float p = probs[idx * C + label];
            p = fmaxf(p, EPSILON);
            local_loss = -__logf(p);
        }
    }

    float block_sum = softmax_block_reduce_sum(local_loss);
    if (threadIdx.x == 0) {
        block_losses[blockIdx.x] = block_sum;
    }
}

__global__ void loss_reduction_kernel(
    const float* __restrict__ block_losses,
    float* __restrict__ total_loss,
    int num_blocks,
    int N
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_losses[i];
    }
    sum = softmax_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *total_loss = sum / static_cast<float>(N);
    }
}

// -------------------------------------------------------------------------
// 3. Combined Softmax + Cross-Entropy Backward Gradient: dZ = (probs - y_onehot) / N
// -------------------------------------------------------------------------
__global__ void softmax_cross_entropy_backward_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ targets,
    float* __restrict__ dZ,
    int N, int C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < C) {
        int target_label = static_cast<int>(targets[row]);
        float y_val = (col == target_label) ? 1.0f : 0.0f;
        dZ[row * C + col] = (probs[row * C + col] - y_val) / static_cast<float>(N);
    }
}

// -------------------------------------------------------------------------
// 4. Argmax Kernel for Class Prediction
// -------------------------------------------------------------------------
__global__ void argmax_kernel(
    const float* __restrict__ probs,
    int* __restrict__ preds,
    int N, int C
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N) {
        float max_p = -1e30f;
        int max_c = 0;
        for (int c = 0; c < C; ++c) {
            float p = probs[row * C + c];
            if (p > max_p) {
                max_p = p;
                max_c = c;
            }
        }
        preds[row] = max_c;
    }
}

// -------------------------------------------------------------------------
// Host Launch Functions
// -------------------------------------------------------------------------
void launch_softmax_forward(
    const float* d_logits,
    float* d_probs,
    int N, int C,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_forward_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_probs, N, C);
}

float launch_cross_entropy_loss(
    const float* d_probs,
    const float* d_targets,
    float* d_block_losses,
    float* d_total_loss,
    int N, int C,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads, 0, stream>>>(
        d_probs, d_targets, d_block_losses, N, C
    );

    loss_reduction_kernel<<<1, 256, 0, stream>>>(
        d_block_losses, d_total_loss, blocks, N
    );

    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_loss;
}

void launch_softmax_cross_entropy_backward(
    const float* d_probs,
    const float* d_targets,
    float* d_dZ,
    int N, int C,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((C + 15) / 16, (N + 15) / 16);
    softmax_cross_entropy_backward_kernel<<<grid, block, 0, stream>>>(
        d_probs, d_targets, d_dZ, N, C
    );
}

void launch_argmax(
    const float* d_probs,
    int* d_preds,
    int N, int C,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    argmax_kernel<<<blocks, threads, 0, stream>>>(d_probs, d_preds, N, C);
}

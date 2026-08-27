#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define WARP_SIZE 32
#define EPSILON 1e-7f

// -------------------------------------------------------------------------
// Device Helper Functions: Warp & Block Reductions
// -------------------------------------------------------------------------
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float sum = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
    }
    return sum;
}

// -------------------------------------------------------------------------
// 1. Forward Kernel: z = X * w + b, y_hat = sigmoid(z)
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
        float z = *b;
        #pragma unroll 4
        for (int j = 0; j < D; ++j) {
            z += X[idx * D + j] * w[j];
        }
        y_hat[idx] = 1.0f / (1.0f + __expf(-z));
    }
}

// -------------------------------------------------------------------------
// 2. BCE Loss Kernel with Reduction
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
        p = fminf(fmaxf(p, EPSILON), 1.0f - EPSILON);
        local_loss = -(target * __logf(p) + (1.0f - target) * __logf(1.0f - p));
    }

    float block_sum = block_reduce_sum(local_loss);

    if (threadIdx.x == 0) {
        block_losses[blockIdx.x] = block_sum;
    }
}

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
// 3. Backward Kernel: Analytical Gradients
// grad_w = (1/N) * X^T * (y_hat - y)
// grad_b = (1/N) * sum(y_hat - y)
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
        for (int i = tid; i < N; i += stride) {
            float err = y_hat[i] - y[i];
            thread_sum += err * X[i * D + feature_idx];
        }
        float block_sum = block_reduce_sum(thread_sum);
        if (tid == 0) {
            grad_w[feature_idx] = block_sum / static_cast<float>(N);
        }
    } else if (feature_idx == D) {
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
// 4. SGD In-Place Update Kernel
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
// Host Interface Functions (Exposed to Python via PyTorch C++ Extension)
// =========================================================================

torch::Tensor forward_cuda(
    torch::Tensor X,
    torch::Tensor w,
    torch::Tensor b
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    int N = X.size(0);
    int D = X.size(1);

    auto y_hat = torch::empty({N}, X.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y_hat.data_ptr<float>(),
        N, D
    );

    return y_hat;
}

torch::Tensor bce_loss_cuda(
    torch::Tensor y_hat,
    torch::Tensor y
) {
    TORCH_CHECK(y_hat.is_cuda() && y.is_cuda(), "Tensors must be on CUDA");
    int N = y_hat.size(0);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    auto block_losses = torch::empty({blocks}, y_hat.options());
    auto total_loss   = torch::empty({1}, y_hat.options());

    bce_loss_kernel<<<blocks, threads>>>(
        y_hat.data_ptr<float>(),
        y.data_ptr<float>(),
        block_losses.data_ptr<float>(),
        N
    );

    final_loss_reduction_kernel<<<1, 256>>>(
        block_losses.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        blocks, N
    );

    return total_loss;
}

std::vector<torch::Tensor> backward_cuda(
    torch::Tensor X,
    torch::Tensor y_hat,
    torch::Tensor y
) {
    TORCH_CHECK(X.is_cuda() && y_hat.is_cuda() && y.is_cuda(), "Tensors must be on CUDA");
    int N = X.size(0);
    int D = X.size(1);

    auto grad_w = torch::empty({D}, X.options());
    auto grad_b = torch::empty({1}, X.options());

    backward_kernel<<<D + 1, 256>>>(
        X.data_ptr<float>(),
        y_hat.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        N, D
    );

    return {grad_w, grad_b};
}

void sgd_step_cuda(
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor grad_w,
    torch::Tensor grad_b,
    float lr
) {
    int D = w.size(0);
    const int threads = 256;
    const int blocks = (D + threads - 1) / threads;

    sgd_update_kernel<<<blocks, threads>>>(
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_b.data_ptr<float>(),
        lr, D
    );
}

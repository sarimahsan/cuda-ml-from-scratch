#ifndef OPTIMIZERS_CUH
#define OPTIMIZERS_CUH

#include "../../00_common/include/cuda_utils.cuh"

// SGD with Momentum: v = momentum * v + grad; param = param - lr * v
void sgd_momentum_update(
    float* d_param,
    float* d_velocity,
    const float* d_grad,
    int size,
    float lr,
    float momentum,
    cudaStream_t stream = 0
);

// Adam: m = beta1 * m + (1 - beta1) * grad; v = beta2 * v + (1 - beta2) * grad^2
// param = param - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
void adam_update(
    float* d_param,
    float* d_m,
    float* d_v,
    const float* d_grad,
    int size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    cudaStream_t stream = 0
);

#endif // OPTIMIZERS_CUH

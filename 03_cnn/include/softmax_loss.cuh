#ifndef SOFTMAX_LOSS_CUH
#define SOFTMAX_LOSS_CUH

#include "../../00_common/include/cuda_utils.cuh"

// Numerically stable Softmax forward
void softmax_forward(
    const float* d_logits,
    float* d_probs,
    int batch_size,
    int num_classes,
    cudaStream_t stream = 0
);

// Categorical Cross-Entropy Loss & Analytical Softmax Error: dZ = (probs - targets) / N
void softmax_cross_entropy_loss_and_grad(
    const float* d_logits,
    const float* d_targets,
    float* d_probs,
    float* d_dZ,
    float* d_loss,
    int batch_size,
    int num_classes,
    cudaStream_t stream = 0
);

#endif // SOFTMAX_LOSS_CUH

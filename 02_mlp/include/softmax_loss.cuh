#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// -------------------------------------------------------------------------
// Softmax & Categorical Cross-Entropy Loss Kernel Declarations
// -------------------------------------------------------------------------

// Softmax Forward: probs = softmax(logits)
void launch_softmax_forward(
    const float* d_logits,
    float* d_probs,
    int N, int C,
    cudaStream_t stream = 0
);

// Categorical Cross-Entropy Loss computation with block & warp reductions
float launch_cross_entropy_loss(
    const float* d_probs,
    const float* d_targets, // float target class indices
    float* d_block_losses,
    float* d_total_loss,
    int N, int C,
    cudaStream_t stream = 0
);

// Softmax + Cross-Entropy Analytical Gradient: dZ = (probs - y_onehot) / N
void launch_softmax_cross_entropy_backward(
    const float* d_probs,
    const float* d_targets,
    float* d_dZ,
    int N, int C,
    cudaStream_t stream = 0
);

// Argmax helper for multi-class classification prediction
void launch_argmax(
    const float* d_probs,
    int* d_preds,
    int N, int C,
    cudaStream_t stream = 0
);

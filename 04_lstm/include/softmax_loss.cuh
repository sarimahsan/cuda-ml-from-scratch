#pragma once

#include "../../00_common/include/cuda_utils.cuh"

// -----------------------------------------------------------------------------
// Numerically stable Sequence Softmax & Cross-Entropy Loss with Warp Reductions
// -----------------------------------------------------------------------------

void launch_sequence_softmax_cross_entropy(
    const float* d_logits, // [Total_Tokens x Vocab_Size]
    const int* d_targets,  // [Total_Tokens]
    float* d_probs,        // [Total_Tokens x Vocab_Size]
    float* d_losses,       // [Total_Tokens]
    float* d_dlogits,      // [Total_Tokens x Vocab_Size] (analytical gradient)
    int total_tokens,
    int vocab_size,
    cudaStream_t stream = 0
);

float compute_mean_loss(
    const float* d_losses,
    int total_tokens,
    cudaStream_t stream = 0
);

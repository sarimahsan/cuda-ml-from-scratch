#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// Softmax & Cross-Entropy Engine
// Logits: (N x C), Probs: (N x C), Targets: (N)
// ============================================================================

// 1. Numerically Stable Warp-Level Softmax Forward Pass
// Each warp processes a row in registers using online max tracking
void softmax_forward(const float* logits, float* probs, int N, int C, cudaStream_t stream = 0);

// 2. Online Safe FlashSoftmax (Computes normalizer and scales in a single pass)
void online_safe_softmax_forward(const float* logits, float* probs, int N, int C, cudaStream_t stream = 0);

// 3. LogSoftmax Forward Pass: out = log(softmax(logits))
void log_softmax_forward(const float* logits, float* log_probs, int N, int C, cudaStream_t stream = 0);

// 4. Softmax Backward Pass: dLogits = dProbs * Probs - Probs * sum(dProbs * Probs)
void softmax_backward(const float* dProbs, const float* probs, float* dLogits,
                      int N, int C, cudaStream_t stream = 0);

// 5. Fused Softmax + Cross-Entropy Loss, Probabilities & Analytical Gradient
// Computes loss = -log(probs[target]), probs = softmax(logits), and dLogits = (probs - 1_{target}) / N
void fused_softmax_cross_entropy_forward_backward(const float* logits, const int64_t* targets,
                                                  float* loss, float* dLogits, float* probs,
                                                  int N, int C, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

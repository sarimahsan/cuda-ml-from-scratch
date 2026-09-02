#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_ml {
namespace kernels {

// ============================================================================
// Vectorized Activations (float4 memory saturation) Forward & Backward
// ============================================================================

enum class ActivationType {
    RELU = 0,
    GELU = 1,
    SIGMOID = 2,
    TANH = 3,
    LEAKY_RELU = 4,
    SILU = 5
};

// 1. Generic Vectorized Activation Forward
void activation_forward(const float* X, float* Y, int N, ActivationType type,
                        float param = 0.01f /* e.g. alpha for LeakyReLU */,
                        cudaStream_t stream = 0);

// 2. Generic Vectorized Activation Backward: dX = dY * dActivation(X or Y)
void activation_backward(const float* dY, const float* X_or_Y, float* dX, int N,
                         ActivationType type, bool is_input_cached = true,
                         float param = 0.01f, cudaStream_t stream = 0);

// 3. Fast Specialized In-Place & Out-of-Place Wrappers
void relu_forward(const float* X, float* Y, int N, cudaStream_t stream = 0);
void relu_backward(const float* dY, const float* X, float* dX, int N, cudaStream_t stream = 0);

void gelu_forward(const float* X, float* Y, int N, cudaStream_t stream = 0);
void gelu_backward(const float* dY, const float* X, float* dX, int N, cudaStream_t stream = 0);

void sigmoid_forward(const float* X, float* Y, int N, cudaStream_t stream = 0);
void sigmoid_backward(const float* dY, const float* Y, float* dX, int N, cudaStream_t stream = 0);

void tanh_forward(const float* X, float* Y, int N, cudaStream_t stream = 0);
void tanh_backward(const float* dY, const float* Y, float* dX, int N, cudaStream_t stream = 0);

void silu_forward(const float* X, float* Y, int N, cudaStream_t stream = 0);
void silu_backward(const float* dY, const float* X, float* dX, int N, cudaStream_t stream = 0);

// 4. SwiGLU: Y = (X1 * sigmoid(X1)) * X2
void swiglu_forward(const float* X1, const float* X2, float* Y, int N, cudaStream_t stream = 0);

} // namespace kernels
} // namespace cuda_ml

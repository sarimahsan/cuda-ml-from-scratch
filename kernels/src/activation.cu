#include "../include/activation.cuh"
#include "../../00_common/include/cuda_utils.cuh"
#include "../../00_common/include/warp_primitives.cuh"
#include <cmath>

namespace cuda_ml {
namespace kernels {

// Vectorized float4 Activation Forward
__global__ void activation_forward_vectorized_kernel(const float* __restrict__ X,
                                                     float* __restrict__ Y,
                                                     int num_vec4,
                                                     ActivationType type,
                                                     float param) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 in_v = load_float4_aligned(X, idx);
        float4 out_v;

        #define APPLY_OP(val)                                                  \
            if (type == ActivationType::RELU) {                                \
                val = fmaxf(0.0f, val);                                        \
            } else if (type == ActivationType::GELU) {                         \
                const float k0 = 0.7978845608028654f;                          \
                const float k1 = 0.044715f;                                    \
                float inner = k0 * (val + k1 * val * val * val);               \
                val = 0.5f * val * (1.0f + tanhf(inner));                      \
            } else if (type == ActivationType::SIGMOID) {                      \
                val = 1.0f / (1.0f + expf(-val));                              \
            } else if (type == ActivationType::TANH) {                         \
                val = tanhf(val);                                              \
            } else if (type == ActivationType::LEAKY_RELU) {                   \
                val = (val > 0.0f) ? val : (param * val);                      \
            } else if (type == ActivationType::SILU) {                         \
                val = val / (1.0f + expf(-val));                               \
            }

        APPLY_OP(in_v.x);
        APPLY_OP(in_v.y);
        APPLY_OP(in_v.z);
        APPLY_OP(in_v.w);
        #undef APPLY_OP

        out_v = in_v;
        store_float4_aligned(Y, idx, out_v);
    }
}

// Vectorized float4 Activation Backward
__global__ void activation_backward_vectorized_kernel(const float* __restrict__ dY,
                                                      const float* __restrict__ X_or_Y,
                                                      float* __restrict__ dX,
                                                      int num_vec4,
                                                      ActivationType type,
                                                      bool is_input_cached,
                                                      float param) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        float4 dy_v = load_float4_aligned(dY, idx);
        float4 xy_v = load_float4_aligned(X_or_Y, idx);
        float4 dx_v;

        #define APPLY_BWD(dy, xy)                                              \
            if (type == ActivationType::RELU) {                                \
                dy = (xy > 0.0f) ? dy : 0.0f;                                  \
            } else if (type == ActivationType::SIGMOID) {                      \
                float s = is_input_cached ? (1.0f / (1.0f + expf(-xy))) : xy;  \
                dy = dy * s * (1.0f - s);                                      \
            } else if (type == ActivationType::TANH) {                         \
                float t = is_input_cached ? tanhf(xy) : xy;                    \
                dy = dy * (1.0f - t * t);                                      \
            } else if (type == ActivationType::LEAKY_RELU) {                   \
                dy = (xy > 0.0f) ? dy : (dy * param);                          \
            } else if (type == ActivationType::GELU) {                         \
                const float k0 = 0.7978845608028654f;                          \
                const float k1 = 0.044715f;                                    \
                float x = xy;                                                  \
                float x3 = x * x * x;                                          \
                float inner = k0 * (x + k1 * x3);                              \
                float tanh_val = tanhf(inner);                                 \
                float sech2 = 1.0f - tanh_val * tanh_val;                      \
                float d_inner = k0 * (1.0f + 3.0f * k1 * x * x);               \
                float grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner; \
                dy = dy * grad;                                                \
            }

        APPLY_BWD(dy_v.x, xy_v.x);
        APPLY_BWD(dy_v.y, xy_v.y);
        APPLY_BWD(dy_v.z, xy_v.z);
        APPLY_BWD(dy_v.w, xy_v.w);
        #undef APPLY_BWD

        dx_v = dy_v;
        store_float4_aligned(dX, idx, dx_v);
    }
}

// Host Launchers
void activation_forward(const float* X, float* Y, int N, ActivationType type,
                        float param, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        activation_forward_vectorized_kernel<<<blocks, threads, 0, stream>>>(
            X, Y, num_vec4, type, param);
    }
}

void activation_backward(const float* dY, const float* X_or_Y, float* dX, int N,
                         ActivationType type, bool is_input_cached,
                         float param, cudaStream_t stream) {
    if (N % 4 == 0) {
        int num_vec4 = N / 4;
        int threads = 256;
        int blocks = (num_vec4 + threads - 1) / threads;
        activation_backward_vectorized_kernel<<<blocks, threads, 0, stream>>>(
            dY, X_or_Y, dX, num_vec4, type, is_input_cached, param);
    }
}

void relu_forward(const float* X, float* Y, int N, cudaStream_t stream) {
    activation_forward(X, Y, N, ActivationType::RELU, 0.0f, stream);
}

void relu_backward(const float* dY, const float* X, float* dX, int N, cudaStream_t stream) {
    activation_backward(dY, X, dX, N, ActivationType::RELU, true, 0.0f, stream);
}

void gelu_forward(const float* X, float* Y, int N, cudaStream_t stream) {
    activation_forward(X, Y, N, ActivationType::GELU, 0.0f, stream);
}

void gelu_backward(const float* dY, const float* X, float* dX, int N, cudaStream_t stream) {
    activation_backward(dY, X, dX, N, ActivationType::GELU, true, 0.0f, stream);
}

void sigmoid_forward(const float* X, float* Y, int N, cudaStream_t stream) {
    activation_forward(X, Y, N, ActivationType::SIGMOID, 0.0f, stream);
}

void sigmoid_backward(const float* dY, const float* Y, float* dX, int N, cudaStream_t stream) {
    activation_backward(dY, Y, dX, N, ActivationType::SIGMOID, false, 0.0f, stream);
}

void tanh_forward(const float* X, float* Y, int N, cudaStream_t stream) {
    activation_forward(X, Y, N, ActivationType::TANH, 0.0f, stream);
}

void tanh_backward(const float* dY, const float* Y, float* dX, int N, cudaStream_t stream) {
    activation_backward(dY, Y, dX, N, ActivationType::TANH, false, 0.0f, stream);
}

void silu_forward(const float* X, float* Y, int N, cudaStream_t stream) {
    activation_forward(X, Y, N, ActivationType::SILU, 0.0f, stream);
}

} // namespace kernels
} // namespace cuda_ml

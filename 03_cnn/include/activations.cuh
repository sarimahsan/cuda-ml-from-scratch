#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include "../../00_common/include/cuda_utils.cuh"

// ReLU
void relu_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream = 0);
void relu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream = 0);

// LeakyReLU
void leaky_relu_forward(const float* d_Z, float* d_A, int size, float alpha = 0.01f, cudaStream_t stream = 0);
void leaky_relu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, float alpha = 0.01f, cudaStream_t stream = 0);

// GELU
void gelu_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream = 0);
void gelu_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream = 0);

// Sigmoid
void sigmoid_forward(const float* d_Z, float* d_A, int size, cudaStream_t stream = 0);
void sigmoid_backward(const float* d_dA, const float* d_Z, float* d_dZ, int size, cudaStream_t stream = 0);

#endif // ACTIVATIONS_CUH

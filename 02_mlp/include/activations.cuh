#pragma once

#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// Activation Function Types & Kernel Declarations
// -------------------------------------------------------------------------

enum class ActivationType {
    RELU,
    SIGMOID,
    GELU,
    LEAKY_RELU
};

// Forward Activation: A = act(Z)
void launch_activation_forward(
    const float* d_Z,
    float* d_A,
    int size,
    ActivationType act_type = ActivationType::RELU,
    cudaStream_t stream = 0
);

// Backward Activation: dZ = dA * act'(Z)
void launch_activation_backward(
    const float* d_dA,
    const float* d_Z,
    float* d_dZ,
    int size,
    ActivationType act_type = ActivationType::RELU,
    cudaStream_t stream = 0
);

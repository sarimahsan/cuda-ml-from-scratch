#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Macro for robust CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Kernel Launch Error: "                         \
                      << cudaGetErrorString(err)                              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

// Accurate GPU Timer using CUDA Events
class GpuTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool is_running;

public:
    GpuTimer() : is_running(false) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        is_running = true;
    }

    float stop(cudaStream_t stream = 0) {
        if (!is_running) return 0.0f;
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        is_running = false;
        return milliseconds;
    }
};

// Memory Allocation Helpers
template <typename T>
T* allocate_device_memory(size_t count) {
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(T)));
    return d_ptr;
}

template <typename T>
void free_device_memory(T* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

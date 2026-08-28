cmake_minimum_required(VERSION 3.18)
project(mlp_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../00_common/include
)

add_executable(mlp_standalone
    src/main.cu
    src/mlp.cu
    src/linear.cu
    src/activations.cu
    src/softmax_loss.cu
    src/optimizers.cu
)

set_target_properties(mlp_standalone PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cmake_minimum_required(VERSION 3.18)
project(01_logistic_regression LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../00_common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(logistic_regression
    src/logistic_regression_kernels.cu
    src/main.cu
)

set_target_properties(logistic_regression PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

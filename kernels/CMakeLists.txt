cmake_minimum_required(VERSION 3.18)
project(cuda_kernels_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../00_common/include
)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(cuda_kernels_engine STATIC ${SOURCES})

set_target_properties(cuda_kernels_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(cuda_kernels_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
)

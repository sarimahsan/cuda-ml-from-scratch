cmake_minimum_required(VERSION 3.18)
project(cuda_ml_models LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/00_common/include)

add_subdirectory(01_logistic_regression)
add_subdirectory(02_mlp)
add_subdirectory(03_cnn)
add_subdirectory(04_lstm)
# Future modules will be added here:
# add_subdirectory(05_attention)
# add_subdirectory(06_transformer)

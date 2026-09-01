cmake_minimum_required(VERSION 3.18)
project(04_lstm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../00_common/include
)

set(CUDA_LSTM_SOURCES
    src/input_gate.cu
    src/forget_gate.cu
    src/cell_candidate_gate.cu
    src/output_gate.cu
    src/cell_state.cu
    src/fused_gates.cu
    src/linear.cu
    src/softmax_loss.cu
    src/optimizers.cu
    src/lstm.cu
)

add_library(cuda_lstm_static STATIC ${CUDA_LSTM_SOURCES})

add_executable(lstm_standalone src/main.cu ${CUDA_LSTM_SOURCES})
target_link_libraries(lstm_standalone PRIVATE ${CUDA_LIBRARIES})

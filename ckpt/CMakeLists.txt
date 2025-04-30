cmake_minimum_required (VERSION 3.25.2)
project(evaluate LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_DIR /usr/local/cuda)
set(TRT_DIR /home/work/file/program/TensorRT-10.9.0.34)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")

# cuda
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)
# tensorrt
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

file(GLOB EVALUATE_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../network_wrapper.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../logger.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mcts.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mcts.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rating.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../evaluator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/evaluate.cpp"
)

find_package(Torch REQUIRED)

add_executable(evaluate ${EVALUATE_SRC_FILES})
target_link_libraries(evaluate ${TORCH_LIBRARIES} cudart nvinfer -lpthread)

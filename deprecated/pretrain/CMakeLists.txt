cmake_minimum_required (VERSION 3.25.2)
project(pretrain LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_DIR /usr/local/cuda)
set(TRT_DIR /home/work/file/program/TensorRT-10.9.0.34)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Og")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")

# cuda
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)
# tensorrt
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

file(GLOB SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../network_wrapper.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../learner.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../network.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../logger.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../replay.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../util.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pretrain.cpp"
)

find_package(Torch REQUIRED)

add_executable(pretrain ${SRC_FILES})

target_link_libraries(pretrain ${TORCH_LIBRARIES} cudart nvinfer)

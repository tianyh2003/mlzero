cmake_minimum_required (VERSION 3.25.2)
project(test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_DIR /usr/local/cuda)
set(TRT_DIR /root/TensorRT-10.9.0.34)

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
    "${CMAKE_CURRENT_SOURCE_DIR}/../logger.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../network.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../mcts_mp.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_network_wrapper.cpp"
)

find_package(Torch REQUIRED)

add_executable(test_network_wrapper ${SRC_FILES})

target_link_libraries(test_network_wrapper ${TORCH_LIBRARIES} cudart nvinfer)

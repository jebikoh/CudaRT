cmake_minimum_required(VERSION 3.28)
project(CudaRT CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")

add_executable(CudaRT main.cu)

set_target_properties(CudaRT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

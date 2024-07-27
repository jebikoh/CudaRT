cmake_minimum_required(VERSION 3.28)
project(CudaRT CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(USE_CUDA ON CACHE BOOL "Enable CUDA support in jtxlib" FORCE)
add_subdirectory(ext/jtxlib)

add_executable(CudaRT src/main.cu
        src/hittable.hpp
        src/rt.hpp
        src/interval.hpp
        src/sphere.hpp
        src/hittablelist.hpp
        src/camera.hpp
)

set(CUDA_ARCH "75" CACHE STRING "CUDA architecture (e.g. 75 for SM 7.5)")
set_target_properties(CudaRT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH}
)

target_link_libraries(CudaRT PRIVATE jtxlib)

target_include_directories(CudaRT
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/jtxlib/src
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/stb
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(CudaRT PRIVATE DEBUG_ASSERT)
elseif (CMAKE_BUILD_TYPE STREQUAL "Test")
    target_compile_definitions(CudaRT PRIVATE TEST_ASSERT)
endif ()
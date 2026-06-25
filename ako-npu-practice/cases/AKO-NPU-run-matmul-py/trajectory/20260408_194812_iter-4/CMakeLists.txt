# ============================================================================
# CMake build config - matmul_leakyrelu Ascend C Kernel
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(matmul_leakyrelu_custom LANGUAGES ASC CXX)

add_executable(matmul_leakyrelu_custom matmul_leakyrelu.asc)

target_link_libraries(matmul_leakyrelu_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(matmul_leakyrelu_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# NPU architecture: dav-2201 for Atlas A2/Ascend910B
target_compile_options(matmul_leakyrelu_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

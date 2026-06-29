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

# big-matmul experiment: shape & cores passed in via -DMM_*  (default = original case)
set(MM_M     1024 CACHE STRING "matmul M")
set(MM_N     640  CACHE STRING "matmul N")
set(MM_K     256  CACHE STRING "matmul K")
set(MM_CORES 1    CACHE STRING "SetDim core count")
set(MM_BLOCKS 1   CACHE STRING "kernel launch grid dim")
set(MM_BASEM 256  CACHE STRING "tiling baseM")
set(MM_BASEN 128  CACHE STRING "tiling baseN")
target_compile_definitions(matmul_leakyrelu_custom PRIVATE
    MM_M=${MM_M} MM_N=${MM_N} MM_K=${MM_K}
    MM_CORES=${MM_CORES} MM_BLOCKS=${MM_BLOCKS} MM_BASEM=${MM_BASEM} MM_BASEN=${MM_BASEN}
)

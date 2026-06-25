# ============================================================================
# CMake build config - attention_backward Ascend C Kernel
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(attention_backward_custom LANGUAGES ASC CXX)

add_executable(attention_backward_custom attention_backward.asc)

target_link_libraries(attention_backward_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(attention_backward_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(attention_backward_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-DHAVE_WORKSPACE>
    $<$<COMPILE_LANGUAGE:ASC>:-DHAVE_TILING>
)

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(fused_elwise_custom LANGUAGES ASC CXX)

add_executable(fused_elwise_custom fused_elwise.asc)

target_link_libraries(fused_elwise_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(fused_elwise_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(fused_elwise_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

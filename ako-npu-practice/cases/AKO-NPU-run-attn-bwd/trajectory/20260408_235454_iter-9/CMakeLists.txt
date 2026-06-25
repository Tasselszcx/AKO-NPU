cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(attention_backward LANGUAGES ASC CXX)

add_executable(attention_backward attention_backward.asc)

target_link_libraries(attention_backward PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(attention_backward PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(attention_backward PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

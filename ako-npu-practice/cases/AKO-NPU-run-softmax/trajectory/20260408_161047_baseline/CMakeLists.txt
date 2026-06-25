cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(softmax_custom LANGUAGES ASC CXX)

add_executable(softmax_custom softmax.asc)

target_link_libraries(softmax_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(softmax_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(softmax_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

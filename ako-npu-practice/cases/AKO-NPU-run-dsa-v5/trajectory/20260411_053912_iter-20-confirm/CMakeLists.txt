cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(dsa_indexer LANGUAGES ASC CXX)

add_executable(dsa_indexer dsa_indexer.asc)

target_link_libraries(dsa_indexer PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(dsa_indexer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# NPU architecture: dav-2201 = Ascend910B (A3)
target_compile_options(dsa_indexer PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

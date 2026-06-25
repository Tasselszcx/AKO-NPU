cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(vae_residual_block LANGUAGES ASC CXX)

add_executable(vae_residual_block vae_residual_block.asc)

target_link_libraries(vae_residual_block PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(vae_residual_block PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(vae_residual_block PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

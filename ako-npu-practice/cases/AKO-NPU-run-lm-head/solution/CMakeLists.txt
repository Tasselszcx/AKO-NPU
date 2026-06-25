cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(lm_head_projection LANGUAGES ASC CXX)

add_executable(demo
    lm_head_projection.asc
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
    dl
)

# NPU architecture: Atlas A2 (910B3) = dav-2201
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

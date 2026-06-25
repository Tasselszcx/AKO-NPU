cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(matmul_leakyrelu LANGUAGES ASC CXX)

add_executable(matmul_leakyrelu
    matmul_leakyrelu.asc
)

target_link_libraries(matmul_leakyrelu PRIVATE
    tiling_api
    register
    platform
    m
    dl
)

target_include_directories(matmul_leakyrelu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(matmul_leakyrelu PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

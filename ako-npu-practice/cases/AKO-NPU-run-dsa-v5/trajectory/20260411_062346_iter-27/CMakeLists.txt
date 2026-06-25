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

# Override host_compile_options_list to add -O3 and AVX512 for host-side SIMD auto-vectorization
# The CMakeASCInformation.cmake wraps host options in -Xhost-start ... -Xhost-end
set(host_compile_options_list -fPIC -O3 -march=native -ffast-math)
list(JOIN host_compile_options_list " " HOST_OPTIONS_LIST)
set(CMAKE_ASC_COMPILE_OBJECT "<CMAKE_ASC_COMPILER> ${CMAKE_COMPILE_AS_ASC_FLAG} <DEFINES> <INCLUDES> -Xaicore-start -O3 -std=c++17 -Xaicore-end -Xhost-start ${HOST_OPTIONS_LIST} -Xhost-end <FLAGS> -o <OBJECT> -c -x asc <SOURCE>")

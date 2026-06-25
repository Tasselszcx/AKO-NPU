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

# dav-2201 (Atlas A2)
target_compile_options(attention_backward PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

# HAVE_WORKSPACE + HAVE_TILING: framework handles system workspace automatically.
# The second-to-last param (sysWs) is treated as workspace, last param (tilingGm) as tiling.
# User workspace (userWs) is passed as a regular parameter before sysWs.
ascendc_compile_definitions(attention_backward PRIVATE
    HAVE_WORKSPACE
    HAVE_TILING
)

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(attention_backward LANGUAGES ASC CXX)

add_executable(attention_backward attention_backward.asc)

# Find ASCEND toolkit path for aclnn headers and libs
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

target_link_libraries(attention_backward PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    nnopbase
    opapi
)

target_include_directories(attention_backward PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME}/x86_64-linux/include
)

target_link_directories(attention_backward PRIVATE
    ${ASCEND_HOME}/x86_64-linux/lib64
)

target_compile_options(attention_backward PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

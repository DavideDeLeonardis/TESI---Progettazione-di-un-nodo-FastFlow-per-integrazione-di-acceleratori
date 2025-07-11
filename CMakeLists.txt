cmake_minimum_required(VERSION 3.18)
project(Tesi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/fastflow
)

add_executable(tesi-cpu-baseline-exec
    src/main.cpp
    src/ff_node_acc_t.cpp
    src/ff_fastflow_patch.cpp
)

enable_testing()
add_test(NAME baseline COMMAND tesi-cpu-baseline-exec 1024)

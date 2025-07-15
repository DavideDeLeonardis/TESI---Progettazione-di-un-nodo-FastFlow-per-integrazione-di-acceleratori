cmake_minimum_required(VERSION 3.18)
project(Tesi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(FF_HEADER_ONLY NO_DEFAULT_MAPPING)


add_executable(tesi-cpu-baseline-exec
    src/main.cpp
    src/ff_node_acc_t.cpp)

target_include_directories(tesi-cpu-baseline-exec PRIVATE
   ${CMAKE_SOURCE_DIR}/external/fastflow
   ${CMAKE_SOURCE_DIR}/include)

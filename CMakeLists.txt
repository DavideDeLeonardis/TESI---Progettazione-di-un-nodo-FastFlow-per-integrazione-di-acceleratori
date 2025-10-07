cmake_minimum_required(VERSION 3.18)
project(Tesi LANGUAGES CXX)

set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(OpenCL REQUIRED)

add_compile_definitions(FF_HEADER_ONLY NO_DEFAULT_MAPPING)

add_executable(tesi-exec
    src/main.cpp
    src/ff_node_acc_t.cpp
    src/CpuAccelerator.cpp
    src/GpuAccelerator.cpp
    src/FpgaAccelerator.cpp)

target_include_directories(tesi-exec PRIVATE
    ${CMAKE_SOURCE_DIR}/external/fastflow
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS})

target_include_directories(tesi-exec PRIVATE ... )
target_link_libraries(tesi-exec PRIVATE OpenCL::OpenCL)

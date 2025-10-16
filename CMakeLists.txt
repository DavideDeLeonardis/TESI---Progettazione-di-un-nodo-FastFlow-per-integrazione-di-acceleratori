cmake_minimum_required(VERSION 3.18)
project(Tesi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")

find_package(OpenCL REQUIRED)

add_compile_definitions(FF_HEADER_ONLY NO_DEFAULT_MAPPING)

add_executable(tesi-exec
    src/main.cpp
    src/ff_node_acc_t.cpp
    src/OclAccelerator.cpp
    src/CpuParallelRunner.cpp
    src/GpuAccelerator.cpp
    src/FpgaAccelerator.cpp
    src/helpers/Helpers.cpp)

    
target_include_directories(tesi-exec PRIVATE
    SYSTEM ${CMAKE_SOURCE_DIR}/external/fastflow
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS})

target_link_libraries(tesi-exec PRIVATE OpenCL::OpenCL)
    
target_compile_definitions(tesi-exec PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_options(tesi-exec PRIVATE -Wno-deprecated-declarations)

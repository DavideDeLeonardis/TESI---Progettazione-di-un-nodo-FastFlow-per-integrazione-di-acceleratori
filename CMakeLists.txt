cmake_minimum_required(VERSION 3.18)

if(APPLE)
    project(Tesi LANGUAGES CXX OBJCXX)
else()
    project(Tesi LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++")
endif()

find_package(OpenCL REQUIRED)

add_compile_definitions(FF_HEADER_ONLY NO_DEFAULT_MAPPING)

# Lista dei file sorgente comuni a tutte le piattaforme.
set(COMMON_SOURCES
    src/main.cpp
    src/cpu_runner/CpuParallelRunner.cpp
    src/accelerator/ff_node_acc_t.cpp
    src/accelerator/BufferManager.cpp
    src/helpers/Helpers.cpp
)

# Aggiunge i file sorgente e le librerie specifiche per ogni piattaforma.
if(APPLE)
    # Su macOS, compiliamo entrambi gli acceleratori GPU.
    list(APPEND COMMON_SOURCES 
        src/accelerator/Gpu_OpenCL_Accelerator.cpp
        src/accelerator/Gpu_Metal_Accelerator.mm
    )
    # Linkiamo i framework di sistema necessari.
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    set(PLATFORM_LIBS ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
else()
    # Su Linux, compiliamo solo l'acceleratore FPGA e linkiamo la libreria filesystem.
    list(APPEND COMMON_SOURCES src/accelerator/FpgaAccelerator.cpp)
    set(PLATFORM_LIBS stdc++fs)
endif()

add_executable(tesi-exec ${COMMON_SOURCES})

# Specifica le directory dove il compilatore deve cercare gli .hpp
target_include_directories(tesi-exec PRIVATE
    SYSTEM ${CMAKE_SOURCE_DIR}/external/fastflow
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS})

# Linka le librerie comuni e quelle specifiche della piattaforma.
target_link_libraries(tesi-exec PRIVATE 
    OpenCL::OpenCL
    ${PLATFORM_LIBS}
)
    
target_compile_definitions(tesi-exec PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_options(tesi-exec PRIVATE -Wno-deprecated-declarations)

# Diciamo a CMake di trattare il file .mm come Objective-C++ e di attivare 
# ARC (gestione automatica della memoria).
if(APPLE)
    set_source_files_properties(src/accelerator/Gpu_Metal_Accelerator.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()
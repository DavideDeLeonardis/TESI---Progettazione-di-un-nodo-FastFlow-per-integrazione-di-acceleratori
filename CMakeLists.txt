cmake_minimum_required(VERSION 3.18)
project(Tesi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Trova il pacchetto OpenCL (necessario per la versione GPU)
find_package(OpenCL REQUIRED)

add_compile_definitions(FF_HEADER_ONLY NO_DEFAULT_MAPPING)

# Cambiamo il nome dell'eseguibile in tesi-gpu-exec
add_executable(tesi-gpu-exec
    src/main.cpp
    src/ff_node_acc_t.cpp
    src/GpuAccelerator.cpp)

# Aggiungiamo gli include per OpenCL
target_include_directories(tesi-gpu-exec PRIVATE
    ${CMAKE_SOURCE_DIR}/external/fastflow
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS})

# Linkiamo la libreria OpenCL
target_link_libraries(tesi-gpu-exec PRIVATE OpenCL::OpenCL)
#pragma once
#include <chrono>
#include <cstddef>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Task di calcolo
struct Task {
   int *a, *b, *c; // Puntatori ai vettori di input/output
   size_t n;       // Dimensione dei vettori

   size_t id{0};         // ID del task
   size_t buffer_idx{0}; // Index del buffer set che il task sta usando

   // Ultimo evento OpenCL generato (usato con GPU_openCL e FPGA).
   cl_event event{nullptr};

   // Handle generico per la sincronizzazione con GPU_Metal.
   void *sync_handle{nullptr};

   // Tempo di arrivo del task nel nodo.
   std::chrono::steady_clock::time_point arrival_time;
};

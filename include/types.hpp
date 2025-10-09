#pragma once
#include <cstddef>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Rappresenta un task di calcolo
struct Task {
   int *a, *b, *c; // Puntatori ai vettori di input/output
   size_t n;       // Dimensione dei vettori

   size_t id{0};            // ID del task
   size_t buffer_idx{0};    // Index del buffer set che il task sta usando
   cl_event event{nullptr}; // Ultimo evento OpenCL generato
};

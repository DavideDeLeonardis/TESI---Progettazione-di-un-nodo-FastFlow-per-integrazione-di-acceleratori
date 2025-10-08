#pragma once

#include "IAccelerator.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Gestisce l'offloading del calcolo su una GPU tramite OpenCL.
class GpuAccelerator : public IAccelerator {
 public:
   GpuAccelerator();
   ~GpuAccelerator() override;

   bool initialize() override;
   void execute(void *generic_task, long long &computed_ns) override;

 private:
   // Risorse OpenCL
   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr}; // Il programma OpenCL (sorgente compilato)
   cl_kernel kernel_{nullptr};

   // Buffer persistenti sulla memoria del device
   cl_mem bufferA{nullptr};
   cl_mem bufferB{nullptr};
   cl_mem bufferC{nullptr};

   // Dimensione in byte dei buffer attualmente allocati
   size_t allocated_size_bytes_{0};
};
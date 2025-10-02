// File: src/GpuAccelerator.hpp

#pragma once

#include "IAccelerator.hpp" // Dobbiamo includere l'interfaccia che stiamo implementando

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// GpuAccelerator è una classe concreta che gestisce l'offloading su una GPU
// tramite OpenCL. Eredita pubblicamente da IAccelerator.
class GpuAccelerator : public IAccelerator {
 public:
   GpuAccelerator();
   // Ora questo 'override' è valido perché ~IAccelerator() è virtual.
   ~GpuAccelerator() override;

   bool initialize() override;
   void execute(void *generic_task, long long &computed_us) override;

 private:
   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr};
   cl_kernel kernel_{nullptr};
};
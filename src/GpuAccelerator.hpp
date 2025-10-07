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
   void execute(void *generic_task, long long &computed_us) override;

 private:
   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr};
   cl_kernel kernel_{nullptr};
};
#pragma once

#include "IAccelerator.hpp"


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class FpgaAccelerator : public IAccelerator {
 public:
   FpgaAccelerator();
   ~FpgaAccelerator() override;

   // Implementazione dei metodi dell'interfaccia IAccelerator
   bool initialize() override;
   void execute(void *generic_task, long long &computed_us) override;

 private:
   // Membri privati per gestire lo stato di OpenCL
   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr};
   cl_kernel kernel_{nullptr};
};
#pragma once

#include "IAccelerator.hpp"

// Interfaccia per l'acceleratore basato su CPU.
class CpuAccelerator : public IAccelerator {
 public:
   CpuAccelerator();
   ~CpuAccelerator() override;

   bool initialize() override;
   void execute(void *generic_task, long long &computed_us) override;
};
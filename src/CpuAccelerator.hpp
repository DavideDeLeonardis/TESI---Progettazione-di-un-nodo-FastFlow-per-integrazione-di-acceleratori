#pragma once

#include "IAccelerator.hpp"

// Implementazione dell'interfaccia per il calcolo su CPU.
// Nota come non abbia bisogno di membri privati specifici.
class CpuAccelerator : public IAccelerator {
 public:
   CpuAccelerator();
   ~CpuAccelerator() override;

   bool initialize() override;
   void execute(void *generic_task, long long &computed_us) override;
};
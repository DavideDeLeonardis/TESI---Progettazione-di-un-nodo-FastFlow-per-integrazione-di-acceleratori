#pragma once

#include "../include/types.hpp"

// Interfaccia astratta per un acceleratore hardware (CPU, GPU, FPGA)
class IAccelerator {
 public:
   virtual ~IAccelerator() = default;

   // Esegue tutte le operazioni di setup una tantum.
   // (es. trovare il device, creare il contesto OpenCL, compilare il kernel).
   virtual bool initialize() = 0;

   // Esegue un singolo task di calcolo sull'acceleratore.
   virtual void execute(void *generic_task, long long &computed_ns) = 0;

   // Non abbiamo bisogno di un metodo 'cleanup()' esplicito perché
   // sfrutteremo il distruttore delle classi concrete (GpuAccelerator, etc.)
   // per rilasciare le risorse.
};
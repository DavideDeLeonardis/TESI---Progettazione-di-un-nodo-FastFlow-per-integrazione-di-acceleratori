#pragma once

#include "../include/types.hpp" // L'interfaccia deve conoscere la struttura Task

// IAccelerator è una classe base astratta (un'interfaccia) che definisce
// le operazioni comuni per qualsiasi dispositivo di accelerazione.
class IAccelerator {
 public:
   // Un distruttore virtuale è FONDAMENTALE in qualsiasi classe base
   // con funzioni virtuali. Assicura che la pulizia venga eseguita
   // correttamente quando un oggetto derivato (es. GpuAccelerator) viene
   // eliminato tramite un puntatore alla base.
   virtual ~IAccelerator() = default;

   // Esegue tutte le operazioni di setup una tantum.
   // (es. trovare il device, creare il contesto OpenCL, compilare il kernel).
   // Restituisce 'true' se l'inizializzazione ha successo, 'false' altrimenti.
   virtual bool initialize() = 0;

   // Esegue un singolo task di calcolo sull'acceleratore.
   // Prende in input il puntatore al task e restituisce il tempo di calcolo
   // (in microsecondi) tramite un parametro passato per riferimento.
   virtual void execute(void *generic_task, long long &computed_us) = 0;

   // Nota: Non abbiamo bisogno di un metodo 'cleanup()' esplicito perché
   // sfrutteremo il distruttore delle classi concrete (GpuAccelerator, etc.)
   // per rilasciare le risorse.
};
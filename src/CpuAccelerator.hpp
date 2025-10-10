#pragma once

#include "IAccelerator.hpp"

/**
 * @brief Implementazione dell'acceleratore per la CPU.
 *
 * La computazione usando solo la CPU non è parallelizzata, ma sequenziale, in
 * quanto l'host e il device sono la stessa entità. La funzione
 * get_results_blocking qui non delega nulla, ma esegue lei stessa l'intero
 * calcolo. Il downloaderLoop è l'unico che lavora e si blocca, deve finire
 * completamente il Task N prima di prendere in carico il Task N+1.
 * CpuAccelerator è solo una simulazione dell'interfaccia a pipeline.
 */
class CpuAccelerator : public IAccelerator {
 public:
   CpuAccelerator();
   ~CpuAccelerator() override;

   bool initialize() override;

   size_t acquire_buffer_set() override;
   void release_buffer_set(size_t index) override;
   void send_data_async(void *task_context) override;
   void execute_kernel_async(void *task_context) override;
   void get_results_blocking(void *task_context,
                             long long &computed_ns) override;
};
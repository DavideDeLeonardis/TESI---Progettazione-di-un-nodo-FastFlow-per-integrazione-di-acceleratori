#pragma once

#include "IAccelerator.hpp"

/**
 * @brief Implementazione dell'acceleratore per la CPU.
 *
 * Poiché la CPU non ha un modello di esecuzione asincrona come GPU/FPGA,
 * simuliamo l'interfaccia a pipeline, PER ADESSO.
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
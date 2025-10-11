#pragma once

#include "IAccelerator.hpp"
#include <mutex>
#include <queue>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @brief Implementazione di IAccelerator che gestisce l'offloading su FPGA.
 */
class FpgaAccelerator : public IAccelerator {
 public:
   FpgaAccelerator();
   ~FpgaAccelerator() override;

   // Esegue tutte le operazioni di setup una volta sola.
   bool initialize() override;

   // Implementazione dei metodi di IAccelerator.
   size_t acquire_buffer_set() override;
   void release_buffer_set(size_t index) override;
   void send_data_async(void *task_context) override;
   void execute_kernel_async(void *task_context) override;
   void get_results_blocking(void *task_context,
                             long long &computed_ns) override;

 private:
   cl_context context_{nullptr};     // Il contesto OpenCL
   cl_command_queue queue_{nullptr}; // La coda di comandi OpenCL
   cl_program program_{nullptr};     // Il programma OpenCL (kernel compilato)
   cl_kernel kernel_{nullptr};       // Il kernel OpenCL (func da eseguire)

   // Set di buffer, 2 per input e 1 per l'output.
   struct BufferSet {
      cl_mem bufferA{nullptr};
      cl_mem bufferB{nullptr};
      cl_mem bufferC{nullptr};
   };

   // Pool di buffer per il device.
   std::vector<BufferSet> buffer_pool_;
   std::queue<size_t> free_buffer_indices_;
   const size_t POOL_SIZE = 3;
   std::mutex pool_mutex_;

   // Dimensione attualmente allocata per i buffer nel pool.
   size_t allocated_size_bytes_{0};

   // Alloca o rialloca tutti i buffer nel buffer_pool_.
   // Viene chiamata la prima volta o quando un task arriva con una dimensione
   // di dati diversa da quella per cui i buffer sono stati allocati.
   bool reallocate_buffers(size_t required_size_bytes);
};
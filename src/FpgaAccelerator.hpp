#pragma once

#include "IAccelerator.hpp"
#include <condition_variable>
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
 *
 * La pipeline interna al nodo ff_node_acc_t è composta da 2 thread:
 * - Il thread Producer esegue gli stadi di Upload e Execute, utilizzando le
 * funzioni qui dichiarate send_data_to_device() e execute_kernel().
 * - Il thread Consumer esegue lo stadio di Download, utilizzando la
 * funzione qui dichiarata get_results_from_device().
 */
class FpgaAccelerator : public IAccelerator {
 public:
   FpgaAccelerator();
   ~FpgaAccelerator() override;

   // Esegue tutte le operazioni di setup una volta sola (creare contesto,
   // coda comandi, compilare kernel, inizializzare pool buffer).
   bool initialize() override;

   // Metodi per l'acquisizione e il rilascio dei buffer.
   size_t acquire_buffer_set() override;
   void release_buffer_set(size_t index) override;

   // Metoodi utili per i thread della pipeline interna.
   void send_data_to_device(void *task_context) override;
   void execute_kernel(void *task_context) override;
   void get_results_from_device(void *task_context,
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

   // Dati per il pool di buffer nel device e vars per gestione concorrenza.
   std::vector<BufferSet> buffer_pool_;
   std::queue<size_t> free_buffer_indices_;
   const size_t POOL_SIZE = 3;
   std::mutex pool_mutex_;
   std::condition_variable buffer_available_cond_;

   // Dimensione attualmente allocata per i buffer nel pool.
   size_t allocated_size_bytes_{0};

   // Alloca o rialloca tutti i buffer nel buffer_pool_.
   // Viene chiamata la prima volta o quando un task arriva con una dimensione
   // di dati diversa da quella per cui i buffer sono stati allocati.
   bool reallocate_buffers(size_t required_size_bytes);
};
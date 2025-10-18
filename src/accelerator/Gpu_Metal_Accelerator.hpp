#pragma once

#include "IAccelerator.hpp"
#include <memory>
#include <string>

// Forward declaration per la classe MetalBufferManager.
class MetalBufferManager;

/**
 * @brief Implementazione di IAccelerator che gestisce l'offloading su GPU Apple
 * tramite il framework nativo Metal.
 */
class Gpu_Metal_Accelerator : public IAccelerator {
 public:
   Gpu_Metal_Accelerator(const std::string &kernel_path, const std::string &kernel_name);
   ~Gpu_Metal_Accelerator() override;

   // Esegue tutte le operazioni di setup una volta sola (trovare device,
   // creare coda comandi, compilare kernel .metal).
   bool initialize() override;

   // Metodi per l'acquisizione e il rilascio dei buffer.
   size_t acquire_buffer_set() override;
   void release_buffer_set(size_t index) override;

   // Metoodi utili per i thread della pipeline interna.
   void send_data_to_device(void *task_context) override;
   void execute_kernel(void *task_context) override;
   void get_results_from_device(void *task_context, long long &computed_ns) override;

 private:
   // --- Oggetti Metal ---
   void *device_{nullptr};        // Puntatore al device Metal (id<MTLDevice>).
   void *command_queue_{nullptr}; // Puntatore alla coda di comandi (id<MTLCommandQueue>).
   void *library_{nullptr};       // Puntatore alla libreria dei kernel compilati (id<MTLLibrary>).
   void *kernel_function_{nullptr}; // Puntatore alla funzione kernel specifica (id<MTLFunction>).
   void *pipeline_state_{
      nullptr}; // Puntatore allo stato della pipeline di calcolo (id<MTLComputePipelineState>).

   // La logica del pool di buffer è incapsulata in un oggetto specifico per Metal.
   std::unique_ptr<MetalBufferManager> buffer_manager_;

   std::string kernel_path_;
   std::string kernel_name_;
};
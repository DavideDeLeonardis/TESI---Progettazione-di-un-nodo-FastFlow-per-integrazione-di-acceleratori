#pragma once

#include "BufferManager.hpp"
#include "IAccelerator.hpp"
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @brief Implementazione di IAccelerator che gestisce l'offloading su GPU.
 *
 * La pipeline interna al nodo ff_node_acc_t è composta da 2 thread:
 * - Il thread Producer esegue gli stadi di Upload e Execute, utilizzando le
 * funzioni qui dichiarate send_data_to_device() e execute_kernel().
 * - Il thread Consumer esegue lo stadio di Download, utilizzando la
 * funzione qui dichiarata get_results_from_device().
 */
class Gpu_OpenCL_Accelerator : public IAccelerator {
 public:
   Gpu_OpenCL_Accelerator(const std::string &kernel_path,
                  const std::string &kernel_name);
   ~Gpu_OpenCL_Accelerator() override;

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

   // Incapsula la logica per l'acquisizione, il rilascio e la riallocazione dei
   // buffer di memoria sul device.
   std::unique_ptr<BufferManager> buffer_manager_;

   std::string kernel_path_;
   std::string kernel_name_;
};
#pragma once

#include "OclAccelerator.hpp"

/**
 * @brief Implementazione di IAccelerator che gestisce l'offloading su FPGA.
 *
 * La pipeline interna al nodo ff_node_acc_t è composta da 2 thread:
 * - Il thread Producer esegue gli stadi di Upload e Execute, utilizzando le
 * funzioni qui dichiarate send_data_to_device() e execute_kernel().
 * - Il thread Consumer esegue lo stadio di Download, utilizzando la
 * funzione qui dichiarata get_results_from_device().
 */
class FpgaAccelerator : public OclAccelerator {
 public:
   FpgaAccelerator(const std::string &xclbin_path,
                   const std::string &kernel_name);

   // Metodo specifico per l'esecuzione su FPGA.
   void execute_kernel(void *task_context) override;

 protected:
   // Metodi specifici per il setup dell'FPGA.
   cl_device_type getDeviceType() const override;
   bool createProgramAndKernel() override;
};

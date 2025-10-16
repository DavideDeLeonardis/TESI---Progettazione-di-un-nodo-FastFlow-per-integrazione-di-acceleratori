#include "FpgaAccelerator.hpp"
#include <fstream>
#include <iostream>
#include <vector>

FpgaAccelerator::FpgaAccelerator(const std::string &xclbin_path,
                                 const std::string &kernel_name)
    : OclAccelerator(xclbin_path, kernel_name) {}

cl_device_type FpgaAccelerator::getDeviceType() const {
   return CL_DEVICE_TYPE_ACCELERATOR;
}

bool FpgaAccelerator::createProgramAndKernel() {
   cl_int ret;
   cl_device_id device_id = nullptr;
   OCL_CHECK(ret,
             clGetContextInfo(context_, CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id), &device_id, NULL),
             return false);

   // Caricamento del file binario dell'FPGA (.xclbin).
   std::ifstream binaryFile(kernel_path_, std::ios::binary);
   if (!binaryFile.is_open()) {
      std::cerr << "[FpgaAccelerator] Could not open kernel file "
                << kernel_path_ << "\n";
      return false;
   }
   binaryFile.seekg(0, binaryFile.end);
   size_t binarySize = binaryFile.tellg();
   binaryFile.seekg(0, binaryFile.beg);
   std::vector<unsigned char> kernelBinary(binarySize);
   binaryFile.read(reinterpret_cast<char *>(kernelBinary.data()), binarySize);

   const unsigned char *binaries[] = {kernelBinary.data()};
   const size_t binary_sizes[] = {binarySize};

   // Crea il programma con il binario xclbin caricato.
   program_ = clCreateProgramWithBinary(context_, 1, &device_id, binary_sizes,
                                        binaries, NULL, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[FpgaAccelerator] Failed to create program from binary.\n";
      return false;
   }

   // Crea il kernel.
   kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[FpgaAccelerator] Failed to create kernel object for '"
                << kernel_name_ << "'.\n";
      return false;
   }

   std::cerr << "[FpgaAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Stadio 2 (Execute).
 * Imposta gli argomenti del kernel e accoda la sua esecuzione, rilasciando
 * l'evento del completamento del trasferimento dati e ottenendo un nuovo evento
 * che rappresenta il completamento del kernel.
 */
void FpgaAccelerator::execute_kernel(void *task_context) {
   cl_int ret;
   auto *task = static_cast<Task *>(task_context);
   BufferSet &current_buffers = buffer_pool_[task->buffer_idx];
   cl_event previous_event = task->event;

   // Imposta gli argomenti del kernel.
   OCL_CHECK(
      ret, clSetKernelArg(kernel_, 0, sizeof(cl_mem), &current_buffers.bufferA),
      return);
   OCL_CHECK(
      ret, clSetKernelArg(kernel_, 1, sizeof(cl_mem), &current_buffers.bufferB),
      return);
   OCL_CHECK(
      ret, clSetKernelArg(kernel_, 2, sizeof(cl_mem), &current_buffers.bufferC),
      return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(int), &(task->n)), return);

   // Accoda l'esecuzione del kernel.
   OCL_CHECK(ret,
             clEnqueueTask(queue_, kernel_, 1, &previous_event, &task->event),
             return);

   if (previous_event)
      clReleaseEvent(previous_event);
}

#include "GpuAccelerator.hpp"
#include <fstream>
#include <iostream>
#include <vector>

GpuAccelerator::GpuAccelerator(const std::string &kernel_path,
                               const std::string &kernel_name)
    : OclAccelerator(kernel_path, kernel_name) {}

cl_device_type GpuAccelerator::getDeviceType() const {
   return CL_DEVICE_TYPE_GPU;
}

bool GpuAccelerator::createProgramAndKernel() {
   cl_int ret;

   // Legge il kernel OpenCL.
   std::ifstream kernelFile(kernel_path_);
   if (!kernelFile.is_open()) {
      std::cerr << "[GpuAccelerator] Could not open kernel file "
                << kernel_path_ << "\n";
      return false;
   }
   std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            (std::istreambuf_iterator<char>()));
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   // Crea il programma OpenCL.
   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[GpuAccelerator] Failed to create program.\n";
      return false;
   }

   // Compila il programma OpenCL.
   cl_device_id device_id = nullptr; // TODO: Get device id properly
   OCL_CHECK(ret,
             clGetContextInfo(context_, CL_CONTEXT_DEVICES,
                              sizeof(cl_device_id), &device_id, NULL),
             return false);
   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[GpuAccelerator] Kernel compilation failed.\n";
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << "Build Log:\n" << log.data() << "\n";
      return false;
   }

   // Crea l'oggetto kernel.
   kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[GpuAccelerator] Failed to create kernel object for '"
                << kernel_name_ << "'.\n";
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Stadio 2 (Execute).
 * Imposta gli argomenti del kernel e accoda la sua esecuzione, rilasciando
 * l'evento del completamento del trasferimento dati e ottenendo un nuovo evento
 * che rappresenta il completamento del kernel.
 */
void GpuAccelerator::execute_kernel(void *task_context) {
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
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n)),
             return);

   // Accoda l'esecuzione del kernel.
   size_t global_work_size = task->n;
   OCL_CHECK(ret,
             clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                                    NULL, 1, &previous_event, &task->event),
             return);

   if (previous_event)
      clReleaseEvent(previous_event);
}

#include "GpuAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Macro semplificata per il controllo degli errori. Nessuna funzione di
// supporto necessaria.
#define OCL_CHECK(call, on_error_action)                                       \
   do {                                                                        \
      cl_int ret = (call);                                                     \
      if (ret != CL_SUCCESS) {                                                 \
         std::cerr << "OpenCL Error (code " << ret << ") in call `" << #call   \
                   << "` at " << __FILE__ << ":" << __LINE__ << std::endl;     \
         on_error_action;                                                      \
      }                                                                        \
   } while (0)

GpuAccelerator::GpuAccelerator() { std::cerr << "[GpuAccelerator] Created.\n"; }

GpuAccelerator::~GpuAccelerator() {
   std::cerr << "[GpuAccelerator] Cleaning up OpenCL resources...\n";
   if (kernel_)
      clReleaseKernel(kernel_);
   if (program_)
      clReleaseProgram(program_);
   if (queue_)
      clReleaseCommandQueue(queue_);
   if (context_)
      clReleaseContext(context_);
}

bool GpuAccelerator::initialize() {
   std::cerr << "[GpuAccelerator] Initializing...\n";
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;
   cl_uint ret_num_devices;
   cl_uint ret_num_platforms;
   cl_int ret; // Necessario per le funzioni che non restituiscono cl_int

   OCL_CHECK(clGetPlatformIDs(1, &platform_id, &ret_num_platforms),
             return false);
   OCL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                            &ret_num_devices),
             return false);

   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create OpenCL context (code " << ret << ")"
                << std::endl;
      return false;
   }

   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create command queue (code " << ret << ")"
                << std::endl;
      return false;
   }

   std::ifstream kernelFile("vecAdd.cl");
   if (!kernelFile.is_open()) {
      std::cerr << "Error: Could not open kernel file vecAdd.cl\n";
      return false;
   }
   std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            std::istreambuf_iterator<char>());
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create program (code " << ret << ")"
                << std::endl;
      return false;
   }

   // Per clBuildProgram, è utile mantenere il log di build in caso di errore
   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to build program (code " << ret << ")"
                << std::endl;
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << "--- BUILD LOG ---\n"
                << std::string(log.begin(), log.end()) << std::endl;
      return false;
   }

   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create kernel (code " << ret << ")"
                << std::endl;
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

void GpuAccelerator::execute(void *generic_task, long long &computed_us) {
   auto *task = static_cast<Task *>(generic_task);
   cl_int ret;
   size_t buffer_size = sizeof(int) * task->n;

   cl_mem bufferA =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   if (!bufferA) {
      OCL_CHECK(ret, return);
   }
   cl_mem bufferB =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   if (!bufferB) {
      OCL_CHECK(ret, return);
   }
   cl_mem bufferC =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);
   if (!bufferC) {
      OCL_CHECK(ret, return);
   }

   auto t0 = std::chrono::steady_clock::now();

   OCL_CHECK(clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size,
                                  task->a, 0, NULL, NULL),
             return);
   OCL_CHECK(clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size,
                                  task->b, 0, NULL, NULL),
             return);
   OCL_CHECK(clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA), return);
   OCL_CHECK(clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB), return);
   OCL_CHECK(clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC), return);
   OCL_CHECK(clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n)),
             return);

   size_t global_work_size = task->n;
   OCL_CHECK(clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                                    NULL, 0, NULL, NULL),
             return);
   OCL_CHECK(clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size,
                                 task->c, 0, NULL, NULL),
             return);
   OCL_CHECK(clFinish(queue_), return);

   auto t1 = std::chrono::steady_clock::now();
   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
}
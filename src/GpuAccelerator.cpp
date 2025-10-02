#include "GpuAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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
   cl_int ret;
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;
   cl_uint ret_num_devices;
   cl_uint ret_num_platforms;

   ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to get Platform IDs\n";
      return false;
   }
   ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                        &ret_num_devices);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to get Device IDs\n";
      return false;
   }

   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_) {
      std::cerr << "Error: Failed to create OpenCL context\n";
      return false;
   }

   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_) {
      std::cerr << "Error: Failed to create command queue\n";
      return false;
   }

   std::ifstream kernelFile("vecAdd.cl");
   if (!kernelFile.is_open()) {
      std::cerr << "Error: Could not open kernel file vecAdd.cl\n";
      return false;
   }
   std::stringstream ss;
   ss << kernelFile.rdbuf();
   std::string kernelSource = ss.str();
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_) {
      std::cerr << "Error: Failed to create program\n";
      return false;
   }

   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to build program\n";
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << std::string(log.begin(), log.end()) << std::endl;
      return false;
   }

   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create kernel\n";
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

// L'unica modifica è qui: il parametro ora è void*
void GpuAccelerator::execute(void *generic_task, long long &computed_us) {
   // Eseguiamo subito il cast al tipo di task che questa classe sa come gestire
   auto *task = static_cast<Task *>(generic_task);

   cl_int ret;

   // 1. Crea i buffer di memoria sulla GPU
   size_t buffer_size = sizeof(int) * task->n;
   cl_mem bufferA =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferB =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferC =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);

   auto t0 = std::chrono::steady_clock::now();

   // 2. Trasferisci i dati dalla CPU alla GPU (bloccante)
   clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size, task->a, 0,
                        NULL, NULL);
   clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size, task->b, 0,
                        NULL, NULL);

   // 3. Imposta gli argomenti del kernel
   clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA);
   clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB);
   clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC);
   clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n));

   // 4. Esegui il kernel
   size_t global_work_size = task->n;
   clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size, NULL, 0,
                          NULL, NULL);

   // 5. Recupera i risultati dalla GPU alla CPU (bloccante)
   clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size, task->c, 0,
                       NULL, NULL);

   clFinish(queue_);

   auto t1 = std::chrono::steady_clock::now();

   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   // 6. Rilascia i buffer creati per questo task
   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
}
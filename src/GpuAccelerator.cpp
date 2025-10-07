#include "GpuAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

GpuAccelerator::GpuAccelerator() {
   std::cerr << "[GpuAccelerator] Created.\n";
}

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
   std::cerr << "[GpuAccelerator] Destroyed.\n";
}

bool GpuAccelerator::initialize() {
   std::cerr << "[GpuAccelerator] Initializing...\n";
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;
   cl_int ret;

   // Trova la piattaforma e il dispositivo GPU
   clGetPlatformIDs(1, &platform_id, NULL);
   clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

   // Crea il contesto OpenCL
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create OpenCL context.\n";
      return false;
   }

   // --- CORREZIONE: Usa clCreateCommandQueue per compatibilità con OpenCL 1.2
   // ---
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Carica il sorgente del kernel dal file
   std::ifstream kernelFile("kernel/vecAdd.cl");
   if (!kernelFile.is_open()) {
      std::cerr
         << "[ERROR] GpuAccelerator: Could not open kernel file vecAdd.cl\n";
      return false;
   }
   std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            (std::istreambuf_iterator<char>()));
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   // Crea il programma dal sorgente
   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create program.\n";
      return false;
   }

   // Compila il programma
   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Kernel compilation failed.\n";
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << "--- KERNEL BUILD LOG ---\n" << log.data() << "\n";
      return false;
   }

   // Estrai il kernel dal programma compilato
   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create kernel object.\n";
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

void GpuAccelerator::execute(void *generic_task, long long &computed_us) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "[GpuAccelerator] Offloading task with N=" << task->n
             << "...\n";

   cl_int ret;
   size_t buffer_size = sizeof(int) * task->n;

   // 1. Creare i buffer di memoria sul device
   cl_mem bufferA =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferB =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferC =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);

   auto t0 = std::chrono::steady_clock::now();

   // 2. Copiare i dati dall'host al device
   clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size, task->a, 0,
                        NULL, NULL);
   clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size, task->b, 0,
                        NULL, NULL);

   // 3. Impostare gli argomenti del kernel
   clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA);
   clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB);
   clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC);
   clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n));

   // 4. Eseguire il kernel
   size_t global_work_size = task->n;
   clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size, NULL, 0,
                          NULL, NULL);

   // 5. Copiare i risultati dal device all'host
   clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size, task->c, 0,
                       NULL, NULL);

   // 6. Attendere il completamento di tutti i comandi in coda
   clFinish(queue_);

   auto t1 = std::chrono::steady_clock::now();
   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   // 7. Rilasciare le risorse
   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
}
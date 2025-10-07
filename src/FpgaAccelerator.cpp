#include "FpgaAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

FpgaAccelerator::FpgaAccelerator() {
   std::cerr << "[FpgaAccelerator] Created.\n";
}

FpgaAccelerator::~FpgaAccelerator() {
   std::cerr << "[FpgaAccelerator] Cleaning up OpenCL resources...\n";
   if (kernel_)
      clReleaseKernel(kernel_);
   if (program_)
      clReleaseProgram(program_);
   if (queue_)
      clReleaseCommandQueue(queue_);
   if (context_)
      clReleaseContext(context_);
   std::cerr << "[FpgaAccelerator] Destroyed.\n";
}

/**
 * @brief Inizializza l'ambiente per l'uso dell'FPGA.
 *
 * Esegue tutte le operazioni di setup una tantum: trova il dispositivo,
 * crea il contesto e la coda di comandi, legge il sorgente del kernel, carica
 * un .xclbin
 */
bool FpgaAccelerator::initialize() {
   std::cerr << "[FpgaAccelerator] Initializing...\n";
   cl_int ret;
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;

   // Trova una piattaforma OpenCL e un dispositivo di tipo ACCELERATOR
   ret = clGetPlatformIDs(1, &platform_id, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to get Platform IDs.\n";
      return false;
   }
   ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id,
                        NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to get FPGA Device ID.\n";
      return false;
   }

   // Creazione del contesto e della coda di comandi.
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_) {
      std::cerr
         << "[ERROR] FpgaAccelerator: Failed to create OpenCL context.\n";
      return false;
   }
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Caricamento del file binario dell'FPGA (.xclbin).
   std::cerr << "[FpgaAccelerator] Loading FPGA binary (krnl_vadd.xclbin)...\n";
   std::ifstream binaryFile("krnl_vadd.xclbin", std::ios::binary);
   if (!binaryFile.is_open()) {
      std::cerr << "[ERROR] FpgaAccelerator: Could not open kernel file "
                   "krnl_vadd.xclbin.\n";
      return false;
   }
   binaryFile.seekg(0, binaryFile.end);
   size_t binarySize = binaryFile.tellg();
   binaryFile.seekg(0, binaryFile.beg);
   std::vector<unsigned char> kernelBinary(binarySize);
   binaryFile.read(reinterpret_cast<char *>(kernelBinary.data()), binarySize);

   const unsigned char *binaries[] = {kernelBinary.data()};
   const size_t binary_sizes[] = {binarySize};

   // Creazione del programma OpenCL direttamente dal binario.
   program_ = clCreateProgramWithBinary(context_, 1, &device_id, binary_sizes,
                                        binaries, NULL, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr
         << "[ERROR] FpgaAccelerator: Failed to create program from binary.\n";
      return false;
   }
   // NON è necessaria la chiamata a clBuildProgram(), perché il programma è già
   // compilato. Questo rende l'inizializzazione dell'FPGA molto più veloce di
   // quella della GPU.

   // Estrazione dell'handle al kernel.
   kernel_ = clCreateKernel(program_, "krnl_vadd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to create kernel.\n";
      return false;
   }

   std::cerr << "[FpgaAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Esegue un singolo task di somma vettoriale sull'FPGA.
 *
 * @param generic_task Puntatore generico al task da eseguire.
 * @param computed_us Riferimento per restituire il tempo di calcolo.
 */
void FpgaAccelerator::execute(void *generic_task, long long &computed_us) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "[FpgaAccelerator] Offloading task with N=" << task->n
             << "...\n";
   cl_int ret;

   // Creazione buffer e trasferimento dati
   size_t buffer_size = sizeof(int) * task->n;
   cl_mem bufferA =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferB =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferC =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);

   auto t0 = std::chrono::steady_clock::now();

   clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size, task->a, 0,
                        NULL, NULL);
   clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size, task->b, 0,
                        NULL, NULL);

   // Impostazione degli argomenti del kernel.
   int n_as_int = static_cast<int>(task->n);
   clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA);
   clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB);
   clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC);
   clSetKernelArg(kernel_, 3, sizeof(int), &n_as_int);

   // Esecuzione del kernel.
   // Il kernel FPGA è di tipo "single-task": viene lanciato una sola volta
   // (global_work_size = 1). A differenza della GPU, non si lanciano N
   // "thread". Sarà il ciclo for all'interno del circuito hardware a iterare
   // per 'n' elementi.
   size_t global_work_size = 1;
   size_t local_work_size = 1;
   clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                          &local_work_size, 0, NULL, NULL);

   // Recupero risultati, sincronizzazione, calcolo tempo di esecuzione e
   // rilascio risorse
   clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size, task->c, 0,
                       NULL, NULL);
   clFinish(queue_);

   auto t1 = std::chrono::steady_clock::now();
   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
}
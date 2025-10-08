#include "FpgaAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

// Macro per il controllo degli errori OpenCL.
#define OCL_CHECK(err_code, call, on_error_action)                             \
   do {                                                                        \
      err_code = (call);                                                       \
      if (err_code != CL_SUCCESS) {                                            \
         std::cerr << "[ERROR] OpenCL call `" #call "` failed with code "      \
                   << err_code << " at " << __FILE__ << ":" << __LINE__        \
                   << std::endl;                                               \
         on_error_action;                                                      \
      }                                                                        \
   } while (0)

/**
 * @brief Costruttore della classe FpgaAccelerator.
 */
FpgaAccelerator::FpgaAccelerator() {
   std::cerr << "[FpgaAccelerator] Created.\n";
}

/**
 * @brief Distruttore della classe FpgaAccelerator.
 *
 * Si occupa di rilasciare in modo sicuro tutte le risorse OpenCL allocate,
 * ovvero i buffer, il kernel, il programma, la coda di comandi e il contesto.
 */
FpgaAccelerator::~FpgaAccelerator() {
   if (bufferA)
      clReleaseMemObject(bufferA);
   if (bufferB)
      clReleaseMemObject(bufferB);
   if (bufferC)
      clReleaseMemObject(bufferC);

   if (kernel_)
      clReleaseKernel(kernel_);
   if (program_)
      clReleaseProgram(program_);
   if (queue_)
      clReleaseCommandQueue(queue_);
   if (context_)
      clReleaseContext(context_);

   std::cerr << "[FpgaAccelerator] Destroyed and OpenCL resources released.\n";
}

/**
 * @brief Inizializza l'ambiente per l'uso dell'FPGA.
 *
 * Esegue tutte le operazioni di setup una tantum: trova il dispositivo,
 * crea il contesto e la coda di comandi, legge il sorgente del kernel, carica
 * un .xclbin
 */
bool FpgaAccelerator::initialize() {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;

   // Trova una piattaforma OpenCL e un dispositivo di tipo ACCELERATOR
   OCL_CHECK(ret, clGetPlatformIDs(1, &platform_id, NULL), return false);
   OCL_CHECK(ret,
             clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1,
                            &device_id, NULL),
             exit(EXIT_FAILURE));

   // Creazione del contesto e della coda di comandi.
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed creating OpenCL context.\n";
      return false;
   }
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Caricamento del file binario dell'FPGA (.xclbin).
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
   // NON è necessaria la chiamata a clBuildProgram(), il programma è già
   // compilato e questo rende l'inizializzazione dell'FPGA molto più veloce di
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
 * @param computed_ns Riferimento per restituire il tempo di calcolo.
 */
void FpgaAccelerator::execute(void *generic_task, long long &computed_ns) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "\n[FpgaAccelerator - START] Offloading task with N=" << task->n
             << "...\n";

   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   size_t required_size_bytes = sizeof(int) * task->n;

   // Se se la dimensione del task corrente è diversa da quella dei buffer già
   // esistenti, rialloca i buffer.
   if (allocated_size_bytes_ != required_size_bytes) {
      std::cerr << "  [FpgaAccelerator - DEBUG] Buffer size mismatch. "
                   "Reallocating buffers for "
                << required_size_bytes << " bytes...\n";

      // Rilascia i vecchi buffer prima di crearne di nuovi
      if (bufferA)
         clReleaseMemObject(bufferA);
      if (bufferB)
         clReleaseMemObject(bufferB);
      if (bufferC)
         clReleaseMemObject(bufferC);

      // Allocazione dei nuovi buffer sulla memoria del device.
      bufferA = clCreateBuffer(context_, CL_MEM_READ_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferA || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] FpgaAccelerator: Failed to create bufferA.\n";
         return;
      }

      bufferB = clCreateBuffer(context_, CL_MEM_READ_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferB || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] FpgaAccelerator: Failed to create bufferB.\n";
         return;
      }

      bufferC = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferC || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] FpgaAccelerator: Failed to create bufferC.\n";
         return;
      }

      allocated_size_bytes_ = required_size_bytes;
   } else {
      std::cerr
         << "  [FpgaAccelerator - DEBUG] Reusing existing device buffers.\n";
   }

   // Inizio misurazione tempo di esecuzione
   auto t0 = std::chrono::steady_clock::now();

   // Trasferimento dati
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0,
                                  required_size_bytes, task->a, 0, NULL, NULL),
             return);
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0,
                                  required_size_bytes, task->b, 0, NULL, NULL),
             return);

   // Impostazione degli argomenti del kernel.
   int n_as_int = static_cast<int>(task->n);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(int), &n_as_int), return);

   // Esecuzione del kernel.
   // Il kernel FPGA è di tipo "single-task": viene lanciato una sola volta
   // (global_work_size = 1). A differenza della GPU, non si lanciano N
   // "thread". Sarà il ciclo for all'interno del circuito hardware a iterare
   // per 'n' elementi.
   size_t global_work_size = 1;
   size_t local_work_size = 1;
   OCL_CHECK(ret,
             clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                                    &local_work_size, 0, NULL, NULL),
             return);

   // Trasferimento dei risultati dalla memoria device alla memoria host.
   OCL_CHECK(ret,
             clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0,
                                 required_size_bytes, task->c, 0, NULL, NULL),
             return);

   // Sincronizzazione.
   OCL_CHECK(ret, clFinish(queue_), return);

   // Fine misurazione tempo di esecuzione
   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[FpgaAccelerator - END] Task execution finished.\n";
}
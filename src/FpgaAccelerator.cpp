#include "FpgaAccelerator.hpp"
#include <chrono>
#include <cstring>
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
 * Aggiunge una barriera di sincronizzazione con clFinish() per garantire che
 * tutti i comandi in coda siano completati, poi rilascia tutte le risorse
 * OpenCL allocate, ovvero i buffer, il kernel, il programma, la coda di comandi
 * e il contesto.
 */
FpgaAccelerator::~FpgaAccelerator() {
   // Barriera di sincronizzazione
   if (queue_)
      clFinish(queue_);

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
             {
                std::cerr << "[FATAL] FPGA Accelerator not found.\n\n";
                exit(EXIT_FAILURE);
             });

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
   // compilato => l'inizializzazione dell'FPGA è molto più veloce di
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

   // Mappatura dei buffer
   int *ptrA =
      (int *)clEnqueueMapBuffer(queue_, bufferA, CL_TRUE, CL_MAP_WRITE, 0,
                                required_size_bytes, 0, NULL, NULL, &ret);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error mapping bufferA\n";
      return;
   }

   int *ptrB =
      (int *)clEnqueueMapBuffer(queue_, bufferB, CL_TRUE, CL_MAP_WRITE, 0,
                                required_size_bytes, 0, NULL, NULL, &ret);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error mapping bufferB\n";
      return;
   }

   // Copia i dati nei buffer mappati
   std::memcpy(ptrA, task->a, required_size_bytes);
   std::memcpy(ptrB, task->b, required_size_bytes);

   // Migrazione dei dati dalla memoria host a quella device.
   cl_mem input_buffers[] = {bufferA, bufferB};
   OCL_CHECK(
      ret,
      clEnqueueMigrateMemObjects(queue_, 2, input_buffers, 0, 0, NULL, NULL),
      return);

   // Impostazione degli argomenti del kernel.
   int n_as_int = static_cast<int>(task->n);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(int), &n_as_int), return);

   // Esecuzione del kernel.
   OCL_CHECK(ret, clEnqueueTask(queue_, kernel_, 0, NULL, NULL), return);

   // Attende il completamento di tutti i comandi in coda (kernel e migrazione
   // dati).
   OCL_CHECK(ret,
             clEnqueueMigrateMemObjects(
                queue_, 1, &bufferC, CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL),
             return);

   // Sincronizzazione
   OCL_CHECK(ret, clFinish(queue_), return);

   // Mappatura del buffer di output per leggere i risultati.
   int *ptrC =
      (int *)clEnqueueMapBuffer(queue_, bufferC, CL_TRUE, CL_MAP_READ, 0,
                                required_size_bytes, 0, NULL, NULL, &ret);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error mapping bufferC\n";
      return;
   }

   // Copia il risultato nel vettore di output del task.
   std::memcpy(task->c, ptrC, required_size_bytes);

   // Rilascio delle mappe dei buffer.
   OCL_CHECK(ret, clEnqueueUnmapMemObject(queue_, bufferA, ptrA, 0, NULL, NULL),
             return);
   OCL_CHECK(ret, clEnqueueUnmapMemObject(queue_, bufferB, ptrB, 0, NULL, NULL),
             return);
   OCL_CHECK(ret, clEnqueueUnmapMemObject(queue_, bufferC, ptrC, 0, NULL, NULL),
             return);

   // Fine misurazione tempo di esecuzione
   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[FpgaAccelerator - END] Task execution finished.\n";
}
#include "GpuAccelerator.hpp"
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
 * @brief Costruttore della classe GpuAccelerator.
 */
GpuAccelerator::GpuAccelerator() { 
   std::cerr << "[GpuAccelerator] Created.\n"; 
}

/**
 * @brief Distruttore della classe GpuAccelerator.
 *
 * Si occupa di rilasciare in modo sicuro tutte le risorse OpenCL allocate,
 * ovvero i buffer, il kernel, il programma, la coda di comandi e il contesto.
 */
GpuAccelerator::~GpuAccelerator() {
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

   std::cerr << "[GpuAccelerator] Destroyed and OpenCL resources released.\n";
}

/**
 * @brief Inizializza l'ambiente OpenCL per l'uso della GPU.
 *
 * Esegue tutte le operazioni di setup una sola volta: trova il dispositivo,
 * crea il contesto e la coda di comandi, legge il sorgente del kernel,
 * lo compila e prepara l'oggetto kernel per l'esecuzione.
 */
bool GpuAccelerator::initialize() {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;

   // Trova una piattaforma OpenCL e un dispositivo di tipo GPU.
   OCL_CHECK(ret, clGetPlatformIDs(1, &platform_id, NULL), return false);
   OCL_CHECK(
      ret, clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL),
      exit(EXIT_FAILURE));

   // Crea un contesto OpenCL.
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create OpenCL context.\n";
      return false;
   }

   // Crea una coda di comandi.
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Legge il codice sorgente del kernel OpenCL da un file esterno.
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

   // Crea un oggetto programma OpenCL a partire dal codice sorgente.
   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create program.\n";
      return false;
   }

   // Compila il programma per il dispositivo target.
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

   // Estrai un handle al kernel compilato ("vecAdd").
   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create kernel object.\n";
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Esegue un singolo task di somma vettoriale sulla GPU.
 *
 * Allocazione della memoria sul device, trasferimento dei dati, impostazione
 * degli argomenti, esecuzione del kernel e recupero dei risultati.
 * @param generic_task Puntatore generico al task da eseguire.
 * @param computed_ns Riferimento per restituire il tempo di calcolo.
 */
void GpuAccelerator::execute(void *generic_task, long long &computed_ns) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "\n[GpuAccelerator - START] Offloading task with N=" << task->n
             << "...\n";

   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   size_t required_size_bytes = sizeof(int) * task->n;

   // Se se la dimensione del task corrente è diversa da quella dei buffer già
   // esistenti, rialloca i buffer.
   if (allocated_size_bytes_ != required_size_bytes) {
      std::cerr << "  [GpuAccelerator - DEBUG] Buffer size mismatch. "
                   "Reallocating buffers for "
                << required_size_bytes << " bytes...\n";

      // Rilascia i vecchi buffer prima di crearne di nuovi
      if (bufferA)
         clReleaseMemObject(bufferA);
      if (bufferB)
         clReleaseMemObject(bufferB);
      if (bufferC)
         clReleaseMemObject(bufferC);

      // Allocazione dei buffer di memoria sulla GPU (device memory).
      bufferA = clCreateBuffer(context_, CL_MEM_READ_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferA || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] Failed to create bufferA. Code: " << ret << "\n";
         return;
      }

      bufferB = clCreateBuffer(context_, CL_MEM_READ_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferB || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] Failed to create bufferB. Code: " << ret << "\n";
         return;
      }

      bufferC = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, required_size_bytes,
                               NULL, &ret);
      if (!bufferC || ret != CL_SUCCESS) {
         std::cerr << "[ERROR] Failed to create bufferC. Code: " << ret << "\n";
         return;
      }

      allocated_size_bytes_ = required_size_bytes;
   } else {
      std::cerr
         << "  [GpuAccelerator - DEBUG] Reusing existing device buffers.\n";
   }

   // Inizio misurazione tempo di esecuzione
   auto t0 = std::chrono::steady_clock::now();

   // Trasferimento dati dalla memoria host (RAM) alla memoria device (VRAM).
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0,
                                  required_size_bytes, task->a, 0, NULL, NULL),
             return);
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0,
                                  required_size_bytes, task->b, 0, NULL, NULL),
             return);

   // Impostazione degli argomenti del kernel.
   OCL_CHECK(ret, clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC), return);
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n)),
             return);

   // Esecuzione del kernel.
   size_t global_work_size = task->n;
   OCL_CHECK(ret,
             clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                                    NULL, 0, NULL, NULL),
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

   std::cerr << "[GpuAccelerator - END] Task execution finished.\n";
}
#include "FpgaAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

/**
 * @file FpgaAccelerator.cpp
 * @brief Implementazione della classe FpgaAccelerator per l'offloading su FPGA.
 */

// Macro per il controllo degli errori OpenCL
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
 * @brief Costruttore.
 */
FpgaAccelerator::FpgaAccelerator() {
   std::cerr << "[FpgaAccelerator] Created.\n";
}

/**
 * @brief Il distruttore si occupa di rilasciare in ordine inverso tutte le
 risorse OpenCL allocate, ovvero i buffer, il kernel, il programma, la coda di
 comandi e il contesto.
 */
FpgaAccelerator::~FpgaAccelerator() {
   // Rilascia tutti i buffer di memoria nel pool
   for (auto &buffer_set : buffer_pool_) {
      if (buffer_set.bufferA)
         clReleaseMemObject(buffer_set.bufferA);
      if (buffer_set.bufferB)
         clReleaseMemObject(buffer_set.bufferB);
      if (buffer_set.bufferC)
         clReleaseMemObject(buffer_set.bufferC);
   }

   // Rilascia gli oggetti OpenCL
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

bool FpgaAccelerator::initialize() {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;

   // Trova una piattaforma OpenCL e un dispositivo di tipo ACCELERATOR.
   OCL_CHECK(ret, clGetPlatformIDs(1, &platform_id, NULL), return false);
   OCL_CHECK(ret,
             clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1,
                            &device_id, NULL),
             {
                std::cerr << "[FATAL] FPGA not found.\n";
                exit(EXIT_FAILURE);
             });

   // Crea un contesto
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed creating OpenCL context.\n";
      return false;
   }

   // Crea la coda di comandi
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

   // Crea il programma con il binario xclbin caricato
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

   // Crea il kernel
   kernel_ = clCreateKernel(program_, "krnl_vadd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to create kernel.\n";
      return false;
   }

   // Inizializza il pool di buffer
   buffer_pool_.resize(POOL_SIZE);
   for (size_t i = 0; i < POOL_SIZE; ++i)
      free_buffer_indices_.push(i);

   std::cerr << "[FpgaAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Helper per allocare o riallocare la memoria per tutti i buffer del
 * pool. Viene invocata al primo utilizzo o quando un task richiede una
 * dimensione di dati differente da quella corrente.
 */
bool FpgaAccelerator::reallocate_buffers(size_t required_size_bytes) {
   std::cerr
      << "  [FpgaAccelerator - DEBUG] Buffer size mismatch or first run. "
      << "Allocating pool buffers for " << required_size_bytes << " bytes.\n";

   // Rilascia eventuali buffer esistenti
   for (auto &buffer_set : buffer_pool_) {
      if (buffer_set.bufferA)
         clReleaseMemObject(buffer_set.bufferA);
      if (buffer_set.bufferB)
         clReleaseMemObject(buffer_set.bufferB);
      if (buffer_set.bufferC)
         clReleaseMemObject(buffer_set.bufferC);
   }

   // Alloca nuovi buffer
   cl_int ret;
   for (size_t i = 0; i < POOL_SIZE; ++i) {
      buffer_pool_[i].bufferA = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                               required_size_bytes, NULL, &ret);
      buffer_pool_[i].bufferB = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                               required_size_bytes, NULL, &ret);
      buffer_pool_[i].bufferC = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                               required_size_bytes, NULL, &ret);
      if (ret != CL_SUCCESS) {
         std::cerr
            << "[ERROR] FpgaAccelerator: Failed to allocate buffer pool.\n";
         return false;
      }
   }

   allocated_size_bytes_ = required_size_bytes;
   return true;
}

/**
 * @brief Acquisisce un indice di buffer dal pool. Se nessun buffer è
 * disponibile, attende in modo non bloccante.
 */
size_t FpgaAccelerator::acquire_buffer_set() {
   std::unique_lock<std::mutex> lock(pool_mutex_);

   // Attende finché non c'è un buffer libero
   while (free_buffer_indices_.empty()) {
      lock.unlock();
      std::this_thread::yield();
      lock.lock();
   }

   // Estrae e restituisce l'indice del buffer libero
   size_t index = free_buffer_indices_.front();
   free_buffer_indices_.pop();
   return index;
}

/**
 * @brief Rilascia un indice di buffer nel pool.
 */
void FpgaAccelerator::release_buffer_set(size_t index) {
   std::lock_guard<std::mutex> lock(pool_mutex_);
   free_buffer_indices_.push(index);
}

/**
 * @brief Stadio 1 della pipeline (Upload).
 * Fa l'upload dei dati di input A e B dall'host alla device memory.
 * L'evento per la sincronizzazione (`task->event`) viene generato solo
 * dall'ultima operazione, garantendo che lo stadio successivo attenda il
 * completamento di entrambi i trasferimenti.
 *
 */
void FpgaAccelerator::send_data_async(void *task_context) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
   auto *task = static_cast<Task *>(task_context);
   BufferSet &current_buffers = buffer_pool_[task->buffer_idx];

   std::cerr << "[FpgaAccelerator - START] Processing task " << task->id
             << " with N=" << task->n << "...\n";

   // Se la dimensione richiesta è diversa da quella allocata, rialloca
   // tutti i buffer del pool.
   size_t required_size_bytes = sizeof(int) * task->n;
   if (allocated_size_bytes_ != required_size_bytes)
      reallocate_buffers(required_size_bytes);

   // Scrivi i due input sulla device memory.
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, current_buffers.bufferA, CL_FALSE, 0,
                                  required_size_bytes, task->a, 0, NULL, NULL),
             return);
   OCL_CHECK(ret,
             clEnqueueWriteBuffer(queue_, current_buffers.bufferB, CL_FALSE, 0,
                                  required_size_bytes, task->b, 0, NULL,
                                  &task->event),
             return);
}

void FpgaAccelerator::execute_kernel_async(void *task_context) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
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

   // Rilascia l'evento precedente.
   if (previous_event)
      clReleaseEvent(previous_event);
}

void FpgaAccelerator::get_results_blocking(void *task_context,
                                           long long &computed_ns) {
   cl_int ret;
   auto *task = static_cast<Task *>(task_context);
   size_t required_size_bytes = sizeof(int) * task->n;
   BufferSet &current_buffers = buffer_pool_[task->buffer_idx];
   cl_event previous_event = task->event;
   auto t0 = std::chrono::steady_clock::now();

   // Recupera i risultati dalla device memory alla memoria host.
   OCL_CHECK(ret,
             clEnqueueReadBuffer(queue_, current_buffers.bufferC, CL_TRUE, 0,
                                 required_size_bytes, task->c, 1,
                                 &previous_event, NULL),
             return);

   // Rilascia l'evento precedente.
   if (previous_event)
      clReleaseEvent(previous_event);
   task->event = nullptr;

   // Calcola il tempo impiegato
   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[FpgaAccelerator - END] Task " << task->id << " finished.\n";
}
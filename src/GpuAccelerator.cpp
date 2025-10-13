#include "GpuAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

/**
 * @brief Implementazione della classe GpuAccelerator per l'offloading su GPU.
 */

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

GpuAccelerator::GpuAccelerator() {}

/**
 * @brief Il distruttore si occupa di rilasciare in ordine inverso tutte le
 risorse OpenCL allocate, ovvero i buffer, il kernel, il programma, la coda di
 comandi e il contesto.
 */
GpuAccelerator::~GpuAccelerator() {
   // Rilascia tutti i buffer di memoria nel pool.
   for (auto &buffer_set : buffer_pool_) {
      if (buffer_set.bufferA)
         clReleaseMemObject(buffer_set.bufferA);
      if (buffer_set.bufferB)
         clReleaseMemObject(buffer_set.bufferB);
      if (buffer_set.bufferC)
         clReleaseMemObject(buffer_set.bufferC);
   }

   // Rilascia gli oggetti OpenCL.
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
 * @brief Esegue tutte le operazioni di setup una volta sola. Trova il
 * dispositivo, crea il contesto, la coda di comandi, legge il sorgente del
 * kernel, lo compila e prepara l'oggetto kernel, inizializza il pool di buffer
 * e la coda degli indici liberi.
 */
bool GpuAccelerator::initialize() {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;

   // Trova una piattaforma OpenCL e un dispositivo di tipo GPU.
   OCL_CHECK(ret, clGetPlatformIDs(1, &platform_id, NULL), return false);
   OCL_CHECK(
      ret, clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL),
      {
         std::cerr << "[FATAL] GPU not found.\n";
         exit(EXIT_FAILURE);
      });

   // Crea un contesto OpenCL.
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create OpenCL context.\n";
      return false;
   }

   // Crea la coda di comandi.
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Legge il kernel OpenCL.
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

   // Crea il programma OpenCL.
   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create program.\n";
      return false;
   }

   // Compila il programma OpenCL.
   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Kernel compilation failed.\n";
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      return false;
   }

   // Crea l'oggetto kernel.
   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create kernel object.\n";
      return false;
   }

   // Inizializza il pool di buffer.
   buffer_pool_.resize(POOL_SIZE);
   for (size_t i = 0; i < POOL_SIZE; ++i)
      free_buffer_indices_.push(i);

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Helper per allocare o riallocare la memoria per tutti i buffer del
 * pool. Viene invocata al primo utilizzo o quando un task richiede una
 * dimensione di dati differente da quella corrente.
 */
bool GpuAccelerator::reallocate_buffers(size_t required_size_bytes) {
   std::cerr << "  [GpuAccelerator - DEBUG] Allocating pool buffers for "
             << required_size_bytes << " bytes\n";

   // Rilascia eventuali buffer esistenti.
   for (auto &buffer_set : buffer_pool_) {
      if (buffer_set.bufferA)
         clReleaseMemObject(buffer_set.bufferA);
      if (buffer_set.bufferB)
         clReleaseMemObject(buffer_set.bufferB);
      if (buffer_set.bufferC)
         clReleaseMemObject(buffer_set.bufferC);
   }

   // Alloca nuovi buffer.
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
            << "[ERROR] GpuAccelerator: Failed to allocate buffer pool.\n";
         return false;
      }
   }

   allocated_size_bytes_ = required_size_bytes;
   return true;
}

/**
 * @brief Acquisisce un indice di buffer dal pool. Se nessun buffer è
 * disponibile per un thread da acquisire, attende in modo non bloccante.
 */
size_t GpuAccelerator::acquire_buffer_set() {
   std::unique_lock<std::mutex> lock(pool_mutex_);

   // Attende finché non c'è un buffer libero.
   buffer_available_cond_.wait(
      lock, [this] { return !free_buffer_indices_.empty(); });

   // Th risvegliato. Estrae e restituisce l'indice del buffer libero.
   size_t index = free_buffer_indices_.front();
   free_buffer_indices_.pop();
   return index;
}

/**
 * @brief Rilascia un indice di buffer nel pool e notifica i thread in attesa.
 */
void GpuAccelerator::release_buffer_set(size_t index) {
   {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      free_buffer_indices_.push(index);
   }

   buffer_available_cond_.notify_one();
}

/**
 * @brief Stadio 1 (Upload).
 * Fa l'upload dei dati di input A e B dall'host alla device memory.
 * L'evento per la sincronizzazione (`task->event`) viene generato solo
 * dall'ultima operazione, garantendo che lo stadio successivo attenda il
 * completamento di entrambi i trasferimenti.
 */
void GpuAccelerator::send_data_to_device(void *task_context) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
   auto *task = static_cast<Task *>(task_context);
   BufferSet &current_buffers = buffer_pool_[task->buffer_idx];

   std::cerr << "[GpuAccelerator - START] Processing task " << task->id
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

/**
 * @brief Stadio 2 (Execute).
 * Imposta gli argomenti del kernel e accoda la sua esecuzione, rilasciando
 * l'evento del completamento del trasferimento dati e ottenendo un nuovo evento
 * che rappresenta il completamento del kernel.
 */
void GpuAccelerator::execute_kernel(void *task_context) {
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
   OCL_CHECK(ret, clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n)),
             return);

   // Accoda l'esecuzione del kernel.
   size_t global_work_size = task->n;
   OCL_CHECK(ret,
             clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size,
                                    NULL, 1, &previous_event, &task->event),
             return);

   // Rilascia l'evento precedente.
   if (previous_event)
      clReleaseEvent(previous_event);
}

/**
 * @brief Stadio 3 (Download).
 * Punto di sincronizzaione. Recupera i risultati dalla device memory alla
 * memoria host, aspettando che l'upload e l'esecuzione del kernel siano
 * completati. È l'unica funzione bloccante della pipeline.
 */
void GpuAccelerator::get_results_from_device(void *task_context,
                                             long long &computed_ns) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
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

   std::cerr << "[GpuAccelerator - END] Task " << task->id << " finished.\n";
}
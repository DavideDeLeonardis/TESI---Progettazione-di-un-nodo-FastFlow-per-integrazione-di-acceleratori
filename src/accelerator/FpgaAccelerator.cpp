#include "FpgaAccelerator.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

/**
 * @brief Implementazione della classe FpgaAccelerator per l'offloading su FPGA.
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

/**
 * @brief Il costruttrore prende in input il nome della funzione kernel e il suo
 * path.
 */
FpgaAccelerator::FpgaAccelerator(const std::string &kernel_path,
                                 const std::string &kernel_name)
    : kernel_path_(kernel_path), kernel_name_(kernel_name) {}

/**
 * @brief Il distruttore si occupa di rilasciare in ordine inverso tutte le
 risorse OpenCL allocate. La pulizia dei buffer è gestita automaticamente dal
 distruttore di buffer_manager_.
 */
FpgaAccelerator::~FpgaAccelerator() {
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
 * @brief Esegue tutte le operazioni di setup una volta sola. Trova il
 * dispositivo, crea il contesto, la coda di comandi, legge il sorgente del
 * kernel, lo compila e prepara l'oggetto kernel, inizializza il pool di buffer
 * e la coda degli indici liberi.
 */
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

   // Chiama il costruttore di BufferManager che iniializza il pool di buffer.
   buffer_manager_ = std::make_unique<BufferManager>(context_);

   // Caricamento del file binario dell'FPGA (.xclbin).
   std::ifstream binaryFile(kernel_path_, std::ios::binary);

   // Controllo che il file sia stato aperto correttamente e che abbia
   // estensione .xclbin, poi lo leggo.
   if (!binaryFile.is_open() ||
       !std::filesystem::is_regular_file(kernel_path_) ||
       kernel_path_.rfind(".xclbin") == std::string::npos) {
      std::cerr << "[ERROR] FpgaAccelerator: Could not open kernel file: "
                << kernel_path_ << "\n";
      exit(EXIT_FAILURE);
   }
   binaryFile.seekg(0, binaryFile.end);
   size_t binarySize = binaryFile.tellg();
   binaryFile.seekg(0, binaryFile.beg);
   std::vector<unsigned char> kernelBinary(binarySize);
   binaryFile.read(reinterpret_cast<char *>(kernelBinary.data()), binarySize);

   const unsigned char *binaries[] = {kernelBinary.data()};
   const size_t binary_sizes[] = {binarySize};

   // Crea il programma con il binario xclbin caricato.
   program_ = clCreateProgramWithBinary(context_, 1, &device_id, binary_sizes,
                                        binaries, NULL, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr
         << "[ERROR] FpgaAccelerator: Failed to create program from binary.\n";
      exit(EXIT_FAILURE);
   }

   // NON è necessaria la chiamata a clBuildProgram(), il programma è già
   // compilato => l'inizializzazione dell'FPGA è molto più veloce di
   // quella della GPU.

   // Crea il kernel.
   kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] FpgaAccelerator: Failed to create kernel.\n";
      exit(EXIT_FAILURE);
   }

   std::cerr << "[FpgaAccelerator] Initialization successful.\n";
   return true;
}

size_t FpgaAccelerator::acquire_buffer_set() {
   return buffer_manager_->acquire_buffer_set();
}

void FpgaAccelerator::release_buffer_set(size_t index) {
   buffer_manager_->release_buffer_set(index);
}

/**
 * @brief Stadio 1 (Upload).
 * Fa l'upload dei dati di input A e B dall'host alla device memory.
 * L'evento per la sincronizzazione (`task->event`) viene generato solo
 * dall'ultima operazione, garantendo che lo stadio successivo attenda il
 * completamento di entrambi i trasferimenti.
 */
void FpgaAccelerator::send_data_to_device(void *task_context) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
   auto *task = static_cast<Task *>(task_context);

   std::cerr << "[FpgaAccelerator - START] Processing task " << task->id
             << " with N=" << task->n << "...\n";

   // Se la dimensione richiesta è diversa da quella allocata, rialloca
   // tutti i buffer del pool e ottieni il set di buffer.
   size_t required_size_bytes = sizeof(int) * task->n;
   buffer_manager_->reallocate_buffers_if_needed(required_size_bytes);
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);

   // Scrive i due input sulla device memory.
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
void FpgaAccelerator::execute_kernel(void *task_context) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL.
   auto *task = static_cast<Task *>(task_context);
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);
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

/**
 * @brief Stadio 3 (Download).
 * Punto di sincronizzaione. Recupera i risultati dalla device memory alla
 * memoria host, aspettando che l'upload e l'esecuzione del kernel siano
 * completati. È l'unica funzione bloccante della pipeline.
 */
void FpgaAccelerator::get_results_from_device(void *task_context,
                                              long long &computed_ns) {
   cl_int ret; // Codice di ritorno delle chiamate OpenCL
   auto *task = static_cast<Task *>(task_context);
   size_t required_size_bytes = sizeof(int) * task->n;
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);
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

   // Calcola il tempo impiegato.
   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[FpgaAccelerator - END] Task " << task->id << " finished.\n";
}
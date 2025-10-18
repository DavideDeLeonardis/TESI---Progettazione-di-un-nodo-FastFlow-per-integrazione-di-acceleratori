#import "Gpu_Metal_Accelerator.hpp"
#include "../common/Task.hpp"
#import <Metal/Metal.h>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <vector>

/**
 *
 * File contente GPU Metal Accelerator e Buffer Manager specifico per Metal.
 *
 */

// =======================================================================
// BufferManager specifico per Metal.
// Gestisce il pool di buffer di memoria sulla GPU.
// =======================================================================
class MetalBufferManager {
 public:
   // Set di buffer, 2 per input e 1 per l'output.
   struct BufferSet {
      id<MTLBuffer> bufferA{nullptr};
      id<MTLBuffer> bufferB{nullptr};
      id<MTLBuffer> bufferC{nullptr};
   };

   /**
    * Costruttore: inizializza il pool di buffer.
    */
   explicit MetalBufferManager(id<MTLDevice> device) : device_(device) {
      buffer_pool_.resize(POOL_SIZE);
      for (size_t i = 0; i < POOL_SIZE; ++i)
         free_buffer_indices_.push(i);
   }

   // ARC di Objective-C rilascia automaticamente gli oggetti MTLBuffer.
   ~MetalBufferManager() {}

   BufferSet &get_buffer_set(size_t index) { return buffer_pool_[index]; }

   bool reallocate_buffers_if_needed(size_t required_size_bytes) {
      if (allocated_size_bytes_ == required_size_bytes)
         return true;

      allocated_size_bytes_ = required_size_bytes;
      // Su Apple Silicon la memoria è condivisa tra CPU e GPU, quindi possiamo
      // accedere agli stessi dati senza copie esplicite sul bus PCIe.
      MTLResourceOptions options = MTLResourceStorageModeShared;

      for (size_t i = 0; i < POOL_SIZE; ++i) {
         buffer_pool_[i].bufferA = [device_ newBufferWithLength:required_size_bytes
                                                        options:options];
         buffer_pool_[i].bufferB = [device_ newBufferWithLength:required_size_bytes
                                                        options:options];
         buffer_pool_[i].bufferC = [device_ newBufferWithLength:required_size_bytes
                                                        options:options];
         if (!buffer_pool_[i].bufferA || !buffer_pool_[i].bufferB || !buffer_pool_[i].bufferC) {
            std::cerr << "[ERROR] MetalBufferManager: Failed to allocate "
                         "buffer pool.\n";
            return false;
         }
      }
      std::cerr << "  [MetalBufferManager - DEBUG] Allocating pool buffers for "
                << required_size_bytes << " bytes\n";

      return true;
   }

   /**
    * Acquisisce un indice di buffer dal pool. Se nessun buffer è
    * disponibile per un thread da acquisire, attende in modo non bloccante.
    */
   size_t acquire_buffer_set() {
      std::unique_lock<std::mutex> lock(pool_mutex_);

      // Attende finché non c'è un buffer libero.
      buffer_available_cond_.wait(lock, [this] { return !free_buffer_indices_.empty(); });

      // Th risvegliato. Estrae e restituisce l'indice del buffer libero.
      size_t index = free_buffer_indices_.front();
      free_buffer_indices_.pop();
      return index;
   }

   /**
    * Rilascia un indice di buffer nel pool e notifica i thread in
    * attesa.
    */
   void release_buffer_set(size_t index) {
      {
         std::lock_guard<std::mutex> lock(pool_mutex_);
         free_buffer_indices_.push(index);
      }
      buffer_available_cond_.notify_one();
   }

 private:
   id<MTLDevice> device_; // Riferimento al device Metal.

   // Dati per il pool di buffer nel device e per la gestione della concorrenza.
   const size_t POOL_SIZE = 3;
   std::vector<BufferSet> buffer_pool_;
   std::queue<size_t> free_buffer_indices_;
   std::mutex pool_mutex_;
   std::condition_variable buffer_available_cond_;

   // Dimensione attualmente allocata per i buffer nel pool.
   size_t allocated_size_bytes_{0};
};

// =======================================================================
// Gpu_Metal_Accelerator
// =======================================================================

Gpu_Metal_Accelerator::Gpu_Metal_Accelerator(const std::string &kernel_path,
                                             const std::string &kernel_name)
    : kernel_path_(kernel_path), kernel_name_(kernel_name) {}

/**
 * Il distruttore usa __bridge_transfer per passare la proprietà dei puntatori C
 * di nuovo ad ARC, che li rilascerà correttamente.
 */
Gpu_Metal_Accelerator::~Gpu_Metal_Accelerator() {
   if (pipeline_state_)
      id<MTLComputePipelineState> pso =
         (__bridge_transfer id<MTLComputePipelineState>)pipeline_state_;
   if (kernel_function_)
      id<MTLFunction> func = (__bridge_transfer id<MTLFunction>)kernel_function_;
   if (library_)
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library_;
   if (command_queue_)
      id<MTLCommandQueue> queue = (__bridge_transfer id<MTLCommandQueue>)command_queue_;
   if (device_)
      id<MTLDevice> dev = (__bridge_transfer id<MTLDevice>)device_;
   buffer_manager_.reset();

   std::cerr << "[Gpu_Metal_Accelerator] Metal resources released.\n";
}

bool Gpu_Metal_Accelerator::initialize() {
   // Trova un device che supporta Metal.
   device_ = (__bridge_retained void *)MTLCreateSystemDefaultDevice();
   if (!device_) {
      std::cerr << "[FATAL] Metal GPU not found.\n";
      exit(EXIT_FAILURE);
   }

   // Crea la coda di comandi.
   id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
   command_queue_ = (__bridge_retained void *)[dev newCommandQueue];
   if (!command_queue_) {
      std::cerr << "[ERROR] MetalAccelerator: Failed to create command queue.\n";
      exit(EXIT_FAILURE);
   }

   // Chiama il costruttore di MetalBufferManager che iniializza il pool di
   // buffer.
   buffer_manager_ = std::make_unique<MetalBufferManager>(dev);

   // Legge il kernel Metal e verifica che il percorso sia un file valido.
   std::ifstream kernelFile(kernel_path_);
   if (!kernelFile.is_open() || !std::filesystem::is_regular_file(kernel_path_)) {
      std::cerr << "[ERROR] MetalAccelerator: Could not open kernel file: " << kernel_path_ << "\n";
      exit(EXIT_FAILURE);
   }
   std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            std::istreambuf_iterator<char>(nullptr));

   NSError *error = nil;

   // Compila il sorgente del kernel .metal in una libreria.
   id<MTLLibrary> lib =
      [dev newLibraryWithSource:[NSString stringWithUTF8String:kernelSource.c_str()]
                        options:nil
                          error:&error];
   if (!lib) {
      std::cerr << "[ERROR] MetalAccelerator: Kernel library compilation "
                   "failed. Check kernel file type.\n";
      exit(EXIT_FAILURE);
   }
   library_ = (__bridge_retained void *)lib;

   // Ottiene la funzione kernel.
   id<MTLFunction> func =
      [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name_.c_str()]];
   if (!func) {
      std::cerr << "[ERROR] MetalAccelerator: Failed to find kernel function '" << kernel_name_
                << "'.\n";
      exit(EXIT_FAILURE);
   }
   kernel_function_ = (__bridge_retained void *)func;

   // Crea lo stato della pipeline di calcolo (oggetto che rappresenta il kernel compilato).
   id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:func error:&error];
   if (!pso) {
      std::cerr << "[ERROR] MetalAccelerator: Failed to create pipeline state "
                   "object.\n";
      if (error) {
         std::cerr << "Reason: " << [[error localizedDescription] UTF8String] << "\n";
      }
      exit(EXIT_FAILURE);
   }
   pipeline_state_ = (__bridge_retained void *)pso;

   std::cerr << "[Gpu_Metal_Accelerator] Initialization successful.\n";
   return true;
}

size_t Gpu_Metal_Accelerator::acquire_buffer_set() { return buffer_manager_->acquire_buffer_set(); }

void Gpu_Metal_Accelerator::release_buffer_set(size_t index) {
   buffer_manager_->release_buffer_set(index);
}

void Gpu_Metal_Accelerator::send_data_to_device(void *task_context) {
   auto *task = static_cast<Task *>(task_context);
   std::cerr << "[Gpu_Metal_Accelerator - START] Processing task " << task->id
             << " with N=" << task->n << "...\n";

   // Se la dimensione richiesta è diversa da quella allocata, rialloca
   // tutti i buffer del pool e ottieni il set di buffer.
   size_t required_size_bytes = sizeof(int) * task->n;
   buffer_manager_->reallocate_buffers_if_needed(required_size_bytes);
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);

   // Grazie alla memoria unificata, copia i dati direttamente.
   memcpy([current_buffers.bufferA contents], task -> a, required_size_bytes);
   memcpy([current_buffers.bufferB contents], task -> b, required_size_bytes);
}

void Gpu_Metal_Accelerator::execute_kernel(void *task_context) {
   auto *task = static_cast<Task *>(task_context);
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);

   // "Prende in prestito" i puntatori agli oggetti Metal per usarli in questa
   // funzione.
   id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
   id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline_state_;

   // Crea un contenitore per i comandi da inviare alla GPU.
   id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
   // Crea un "codificatore" per scrivere i comandi di calcolo.
   id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

   // Imposta il kernel e i suoi argomenti (i buffer).
   [encoder setComputePipelineState:pso];
   [encoder setBuffer:current_buffers.bufferA offset:0 atIndex:0];
   [encoder setBuffer:current_buffers.bufferB offset:0 atIndex:1];
   [encoder setBuffer:current_buffers.bufferC offset:0 atIndex:2];
   unsigned int n_uint = task->n;
   [encoder setBytes:&n_uint length:sizeof(unsigned int) atIndex:3];

   // Definisce la griglia di calcolo (quanti thread lanciare).
   MTLSize grid_size = MTLSizeMake(task->n, 1, 1);
   NSUInteger thread_group_width = [pso maxTotalThreadsPerThreadgroup];
   if (thread_group_width > task->n) {
      thread_group_width = task->n;
   }
   MTLSize thread_group_size = MTLSizeMake(thread_group_width, 1, 1);

   // Accoda il comando di esecuzione del kernel.
   [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];

   // Finalizza la codifica dei comandi.
   [encoder endEncoding];

   // Invia il command buffer alla GPU per l'esecuzione asincrona.
   [command_buffer commit];

   // Salva il command buffer nel task per la sincronizzazione successiva.
   task->sync_handle = (__bridge_retained void *)command_buffer;
}

void Gpu_Metal_Accelerator::get_results_from_device(void *task_context, long long &computed_ns) {
   auto *task = static_cast<Task *>(task_context);

   // Recupera il command buffer dal task e riprende la sua proprietà.
   id<MTLCommandBuffer> command_buffer = (__bridge_transfer id<MTLCommandBuffer>)task->sync_handle;

   auto t0 = std::chrono::steady_clock::now();

   // Attende il completamento del kernel (op. bloccante).
   [command_buffer waitUntilCompleted];
   auto t1 = std::chrono::steady_clock::now();
   computed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   // Copia i risultati indietro nella memoria host.
   size_t required_size_bytes = sizeof(int) * task->n;
   auto &current_buffers = buffer_manager_->get_buffer_set(task->buffer_idx);
   memcpy(task->c, [current_buffers.bufferC contents], required_size_bytes);

   std::cerr << "[Gpu_Metal_Accelerator - END] Task " << task->id << " finished.\n";
}
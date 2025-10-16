#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @brief Gestisce un pool di set di buffer OpenCL. Incapsula la logica per
 * l'acquisizione, il rilascio e la riallocazione dei buffer di memoria sul
 * device.
 */
class BufferManager {
 public:
   explicit BufferManager(cl_context context);
   ~BufferManager();

   // Set di buffer, 2 per input e 1 per l'output.
   struct BufferSet {
      cl_mem bufferA{nullptr};
      cl_mem bufferB{nullptr};
      cl_mem bufferC{nullptr};
   };

   // Metodi per l'acquisizione e il rilascio dei buffer.
   size_t acquire_buffer_set();
   void release_buffer_set(size_t index);

   // Viene chiamata la prima volta o quando un task arriva con una
   // dimensione di dati diversa da quella per cui i buffer sono stati allocati.
   bool reallocate_buffers_if_needed(size_t required_size_bytes);

   // Restituisce un riferimento a un set di buffer specifico.
   BufferSet &get_buffer_set(size_t index);

 private:
   cl_context context_; // Contesto OpenCL per creare i buffer

   // Dati per il pool di buffer nel device e vars per gestione concorrenza.
   std::vector<BufferSet> buffer_pool_;
   std::queue<size_t> free_buffer_indices_;
   const size_t POOL_SIZE = 3;
   std::mutex pool_mutex_;
   std::condition_variable buffer_available_cond_;

   // Dimensione attualmente allocata per i buffer nel pool.
   size_t allocated_size_bytes_{0};
};

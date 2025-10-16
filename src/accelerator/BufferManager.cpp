#include "BufferManager.hpp"
#include <iostream>

/**
 * @brief Costruttore: inizializza il pool di buffer.
 */
BufferManager::BufferManager(cl_context context) : context_(context) {
   buffer_pool_.resize(POOL_SIZE);
   for (size_t i = 0; i < POOL_SIZE; ++i)
      free_buffer_indices_.push(i);
}

/**
 * @brief Distruttore: rilascia tutti i buffer di memoria nel pool.
 */
BufferManager::~BufferManager() {
   for (auto &buffer_set : buffer_pool_) {
      if (buffer_set.bufferA)
         clReleaseMemObject(buffer_set.bufferA);
      if (buffer_set.bufferB)
         clReleaseMemObject(buffer_set.bufferB);
      if (buffer_set.bufferC)
         clReleaseMemObject(buffer_set.bufferC);
   }
}

BufferManager::BufferSet &BufferManager::get_buffer_set(size_t index) {
   return buffer_pool_[index];
}

/**
 * @brief Helper per allocare o riallocare la memoria per tutti i buffer del
 * pool. Viene invocata al primo utilizzo o quando un task richiede una
 * dimensione di dati differente da quella corrente.
 */
bool BufferManager::reallocate_buffers_if_needed(size_t required_size_bytes) {
   if (allocated_size_bytes_ == required_size_bytes)
      return true; // Nessuna riallocazione necessaria

   std::cerr << "  [BufferManager - DEBUG] Allocating pool buffers for "
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
            << "[ERROR] BufferManager: Failed to allocate buffer pool.\n";
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
size_t BufferManager::acquire_buffer_set() {
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
void BufferManager::release_buffer_set(size_t index) {
   {
      std::lock_guard<std::mutex> lock(pool_mutex_);
      free_buffer_indices_.push(index);
   }
   buffer_available_cond_.notify_one();
}

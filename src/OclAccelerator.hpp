#pragma once

#include "IAccelerator.hpp"
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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
 * @brief Classe base astratta per acceleratori OpenCL.
 *
 * Contiene tutta la logica comune per la gestione delle risorse OpenCL,
 * il pool di buffer e la pipeline di offloading. Le classi figlie devono
 * implementare solo le parti specifiche del device (es. caricamento del
 * kernel ed esecuzione).
 */
class OclAccelerator : public IAccelerator {
 public:
   // Il costruttore ora accetta i percorsi per rendere la classe più generica.
   OclAccelerator(std::string kernel_path, std::string kernel_name);
   ~OclAccelerator() override;

   // Implementazioni comuni dell'interfaccia IAccelerator.
   bool initialize() override;
   size_t acquire_buffer_set() override;
   void release_buffer_set(size_t index) override;
   void send_data_to_device(void *task_context) override;
   void get_results_from_device(void *task_context,
                                long long &computed_ns) override;

 protected:
   // Metodi che le classi figlie devono implementare.
   virtual cl_device_type getDeviceType() const = 0;
   virtual bool createProgramAndKernel() = 0;

   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr};
   cl_kernel kernel_{nullptr};

   // Nomi per il file del kernel e la funzione, ora configurabili.
   std::string kernel_path_;
   std::string kernel_name_;

   // Set di buffer, 2 per input e 1 per l'output.
   struct BufferSet {
      cl_mem bufferA{nullptr};
      cl_mem bufferB{nullptr};
      cl_mem bufferC{nullptr};
   };

   // Dati per il pool di buffer nel device e vars per gestione concorrenza.
   std::vector<BufferSet> buffer_pool_;
   std::queue<size_t> free_buffer_indices_;
   const size_t POOL_SIZE = 3;
   std::mutex pool_mutex_;
   std::condition_variable buffer_available_cond_;

   // Dimensione attualmente allocata per i buffer nel pool.
   size_t allocated_size_bytes_{0};

   // Alloca o rialloca tutti i buffer nel buffer_pool_.
   bool reallocate_buffers(size_t required_size_bytes);
};

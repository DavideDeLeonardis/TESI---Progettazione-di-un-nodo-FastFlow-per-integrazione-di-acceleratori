#pragma once

#include "IAccelerator.hpp"
#include <mutex>
#include <queue>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @brief Implementazione di IAccelerator per GPU, usata in una pipeline a 3
 * stadi (upload, execute, download). Gestisce l'offloading del calcolo su una
 * GPU tramite OpenCL.
 */
class GpuAccelerator : public IAccelerator {
 public:
   GpuAccelerator();

   /**
    * @brief Distruttore. Si occupa di rilasciare tutte le risorse OpenCL
    * allocate, quali i buffer, il kernel, il programma, la coda di comandi
    * e il contesto.
    */
   ~GpuAccelerator() override;

   /**
    * @brief Inizializza l'ambiente OpenCL per l'uso della GPU.
    * Trova il dispositivo, crea il contesto, la coda di comandi, legge il
    * sorgente del kernel, lo compila e prepara l'oggetto kernel.
    */
   bool initialize() override;

   // --- Implementazione dell'interfaccia a pipeline ---

   /**
    * @brief Acquisisce un set di buffer libero dal pool del device.
    * @return L'indice del set di buffer acquisito
    */
   size_t acquire_buffer_set() override;

   /**
    * @brief Rilascia un set di buffer, rendendolo nuovamente disponibile nel
    * pool.
    * @param index L'indice del set di buffer da rilasciare.
    */
   void release_buffer_set(size_t index) override;

   /**
    * @brief Stadio 1 (Upload): Accoda il trasferimento dei dati di input
    * dall'host al device in modo asincrono.
    * @param task_context Puntatore a un oggetto Task che contiene i dati e lo
    * stato (incluso l'indice del buffer da usare).
    */
   void send_data_async(void *task_context) override;

   /**
    * @brief Stadio 2 (Execute): Accoda l'esecuzione del kernel OpenCL in modo
    * asincrono. L'esecuzione è dipendente dal completamento del trasferimento
    * dati del medesimo task.
    * @param task_context Puntatore a un oggetto Task che contiene lo stato,
    * incluso l'evento di dipendenza.
    */
   void execute_kernel_async(void *task_context) override;

   /**
    * @brief Stadio 3 (Download): Attende il completamento di tutte le
    * operazioni precedenti per un task e recupera i risultati.
    * Questa è l'unica funzione bloccante della pipeline. Si sincronizza
    * con il completamento del kernel e accoda una lettura bloccante dei dati.
    * @param task_context Puntatore a un oggetto Task.
    * @param computed_ns Riferimento a una variabile dove verrà memorizzato
    * il tempo impiegato per questa fase bloccante (in nanosecondi).
    */
   void get_results_blocking(void *task_context,
                             long long &computed_ns) override;

 private:
   cl_context context_{nullptr};     // Il contesto OpenCL
   cl_command_queue queue_{nullptr}; // La coda di comandi OpenCL
   cl_program program_{nullptr};     // Il programma OpenCL (kernel compilato)
   cl_kernel kernel_{nullptr};       // Il kernel OpenCL (func da eseguire)

   struct BufferSet {
      cl_mem bufferA{nullptr}; // Buffer di input A.
      cl_mem bufferB{nullptr}; // Buffer di input B.
      cl_mem bufferC{nullptr}; // Buffer di output C.
   };

   // Pool di buffer per il device
   std::vector<BufferSet> buffer_pool_;
   static const size_t POOL_SIZE = 3;
   std::mutex pool_mutex_;

   // Coda contenente gli indici dei buffer liberi
   std::queue<size_t> free_buffer_indices_;

   // Dimensione attualmente allocata per i buffer
   size_t allocated_size_bytes_{0};

   // Alloca o rialloca tutti i buffer nel `buffer_pool_`.
   // Viene chiamata la prima volta o quando un task arriva con una dimensione
   // di dati diversa da quella per cui i buffer sono stati allocati.
   bool reallocate_buffers(size_t required_size_bytes);
};
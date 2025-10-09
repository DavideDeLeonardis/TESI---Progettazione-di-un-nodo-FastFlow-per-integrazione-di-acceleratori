#pragma once
#include "../include/types.hpp"

/**
 * @brief Interfaccia per un acceleratore hardware (es. GPU, FPGA).
 *
 * Definisce i 3 stadi della pipeline interna asincrona e la gestione del pool
 * di buffer.
 */
class IAccelerator {
 public:
   virtual ~IAccelerator() = default;

   // Esegue tutte le operazioni di setup una tantum.
   // (es. trovare il device, creare il contesto OpenCL, compilare il kernel).
   virtual bool initialize() = 0;

   /**
    * @brief Acquisisce un set di buffer libero dal pool del device.
    * @return L'indice del set di buffer acquisito.
    */
   virtual size_t acquire_buffer_set() = 0;

   /**
    * @brief Rilascia un set di buffer nel pool del device.
    * @param index L'indice del set da rilasciare.
    */
   virtual void release_buffer_set(size_t index) = 0;

   /**
    * @brief Stadio 1 - Upload: Invia i dati di input dal host al device in modo
    * asincrono.
    * @param task_context Puntatore a un oggetto Task.
    */
   virtual void send_data_async(void *task_context) = 0;

   /**
    * @brief Stadio 2 - Execute: Accoda l'esecuzione del kernel in modo
    * asincrono.
    * @param task_context Puntatore a un oggetto Task.
    */
   virtual void execute_kernel_async(void *task_context) = 0;

   /**
    * @brief Stadio 3 - Download: Attende il completamento del kernel e recupera
    * i risultati dal device all'host.
    * @param task_context Puntatore a un oggetto Task.
    * @param computed_ns Tempo di calcolo.
    */
   virtual void get_results_blocking(void *task_context,
                                     long long &computed_ns) = 0;
};
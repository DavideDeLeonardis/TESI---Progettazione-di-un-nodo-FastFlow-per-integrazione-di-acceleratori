#pragma once

#include "structs/Task.hpp"

/**
 * @brief Interfaccia per un acceleratore hardware (es. GPU, FPGA).
 *
 * Definisce le funzioni per ottenere e rilasciare i buffer, e le 3 funzioni
 * principali che implementano i due thread della pipeline interna:
 * - Thread Producer (stadi 1 e 2): send_data_to_device() e
 * execute_kernel().
 * - Thread Consumer (stadio 3): get_results_from_device().
 *
 * ! TODO: Eliminare duplicazione estrema in FpgaAccelerator e GpuAccelerator.
 * ! TODO: Eliminare i nomi hardcoded dei kernel (.xclbin, .cl).
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
    * @brief Stadio 1 - Upload: Invia i dati di input dall'host al device.
    * @param task_context Puntatore a un oggetto Task che contiene i dati e lo
    * stato (incluso l'indice del buffer da usare).
    */
   virtual void send_data_to_device(void *task_context) = 0;

   /**
    * @brief Stadio 2 - Execute: Accoda l'esecuzione del kernel sul device.
    * Non attende il completamento del kernel.
    * @param task_context Puntatore a un oggetto Task che contiene lo stato,
    * incluso l'evento di dipendenza.
    */
   virtual void execute_kernel(void *task_context) = 0;

   /**
    * @brief Stadio 3 - Download: Attende il completamento di tutte le
    * operazioni precedenti per un task e recupera i risultati dal device
    * all'host. Questa è l'unica funzione bloccante della pipeline. Si
    * sincronizza con il completamento del kernel e accoda il trasferimento
    * dei dati di output all'host.
    * @param task_context Puntatore a un oggetto Task.
    * @param computed_ns Tempo di calcolo effettivo.
    */
   virtual void get_results_from_device(void *task_context,
                                        long long &computed_ns) = 0;
};
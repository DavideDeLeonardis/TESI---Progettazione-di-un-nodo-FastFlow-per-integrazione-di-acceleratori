#pragma once
#include "../include/ff_includes.hpp"
#include "../include/types.hpp"
#include "IAccelerator.hpp"
#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <thread>

/**
 * @brief Nodo FastFlow che orchestra l'offloading su un acceleratore.
 *
 * Implementa una pipeline interna a 3 stadi (Upload, Execute, Download)
 * gestita da 3 thread dedicati per sovrapporre calcolo e trasferimenti dati:
 * 1. Uploader: Prepara i dati e avvia il trasferimento asincrono verso il
 * device.
 * 2. Launcher: Avvia l'esecuzione asincrona del kernel sul device.
 * 3. Downloader: Attende il completamento, recupera i risultati e
 * finalizza il task.
 */
class ff_node_acc_t : public ff_node {
 public:
   explicit ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                          std::promise<size_t> &&count_promise);
   ~ff_node_acc_t() override;

   /**
    * @brief Restituisce il tempo totale di calcolo effettivo (senza overhead
    * di trasferimento dati) in nanosecondi.
    */
   long long getComputeTime_ns() const;

 protected:
   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   // Sentinella usata per segnalare la fine dello stream di dati nelle code
   // interne ai thread.
   static void *const SENTINEL;

   // ---- Loops dei 3 stadi della pipeline interna
   // Prende un task, un buffer e avvia il trasferimento dati asincrono.
   void uploaderLoop();
   // Prende un task con i dati pronti e avvia asincronamente il kernel.
   void launcherLoop();
   // Prende un task eseguito, attende/recupera i risultati e finalizza.
   void downloaderLoop();

   std::unique_ptr<IAccelerator> accelerator_;

   // Code interne Single-Producer/Single-Consumer per la pipeline.
   using TaskQ = uSWSR_Ptr_Buffer;
   // Coda per ricevere i dati dalla pipeline FF esterna.
   TaskQ *inQ_{nullptr};
   // Coda per i task pronti all'esecuzione del kernel.
   TaskQ *kernel_ready_queue_{nullptr};
   // Coda per i task pronti per la lettura dal device all'host.
   TaskQ *readout_ready_queue_{nullptr};

   // Contatori e promise per i risultati.
   std::atomic<long long> computed_ns_{0};
   std::atomic<size_t> tasks_processed_{0};
   std::promise<size_t> count_promise_;

   // I 3 thread per gli stadi della pipeline interna.
   std::thread uploaderTh_, launcherTh_, downloaderTh_;
};
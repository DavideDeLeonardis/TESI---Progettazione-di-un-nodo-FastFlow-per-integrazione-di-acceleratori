#pragma once

#include "../../include/ff_includes.hpp"
#include "../common/BlockingQueue.hpp"
#include "../common/StatsCollector.hpp"
#include "../common/Task.hpp"
#include "IAccelerator.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <thread>

/**
 * @brief Nodo FastFlow che orchestra l'offloading su un acceleratore.
 *
 * Implementa una pipeline interna a 2 stadi gestita da 2 thread:
 * 1. Producer (Upload+Launch): Trasferisce i dati dall'host al device e
 *   avvia l'esecuzione del kernel.
 * 2. Consumer (Download): Trasferisce i risultati dal device all'host.
 *
 * Permette di sovrapporre le operazioni di I/O con il calcolo, nella pipeline
 * il task 'n' è in esecuzione, mentre i dati per 'n+1' vengono caricati e i
 * risultati di 'n-1' vengono scaricati.
 */
class ff_node_acc_t : public ff_node {
 public:
   explicit ff_node_acc_t(IAccelerator *acc, StatsCollector *stats);
   ~ff_node_acc_t() override;

 protected:
   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   // Sentinella usata per segnalare la fine dello stream di dati nelle code
   // interne ai thread.
   static void *const SENTINEL;

   // Loops dei 2 stadi della pipeline interna.
   void producerLoop();
   void consumerLoop();

   // Puntatori all'acceleratore e all'oggetto per le statistiche.
   IAccelerator *accelerator_;
   StatsCollector *stats_;

   // Code per i task in ingresso dalla pipeline FF e per i task pronti per
   // il download dal device all'host.
   BlockingQueue<void *> inQ_;
   BlockingQueue<void *> readyQ_;

   std::thread producerTh_, consumerTh_;
};
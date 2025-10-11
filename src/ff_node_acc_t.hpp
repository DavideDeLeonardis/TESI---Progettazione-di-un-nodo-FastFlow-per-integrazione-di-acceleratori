#pragma once

#include "../include/ff_includes.hpp"
#include "BlockingQueue.hpp"
#include "IAccelerator.hpp"
#include "StatsCollector.hpp"
#include "Task.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <thread>

/**
 * @brief Nodo FastFlow che orchestra l'offloading su un acceleratore.
 *
 * Implementa una pipeline interna a 2 stadi (Producer, Consumer) gestita da 2
 * thread:
 * 1. Producer (Uploader+Launcher): Trasferisce i dati dall'host al device e
 *   avvia l'esecuzione del kernel.
 * 2. Consumer (Downloader): Trasferisce i risultati dal device all'host.
 */
class ff_node_acc_t : public ff_node {
 public:
   explicit ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                          StatsCollector *stats);
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
   std::unique_ptr<IAccelerator> accelerator_;
   StatsCollector *stats_;

   // Code per i task in ingresso dalla pipeline FF e per i task pronti per
   // il download dal device all'host.
   BlockingQueue<void *> inQ_;
   BlockingQueue<void *> readyQ_;

   std::thread producerTh_, consumerTh_;
};
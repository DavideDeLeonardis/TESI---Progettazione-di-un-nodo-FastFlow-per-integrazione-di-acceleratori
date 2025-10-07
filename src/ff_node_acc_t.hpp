#pragma once
#include "../include/types.hpp"
#include "IAccelerator.hpp"
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/all2all.hpp>
#include <ff/graph_utils.hpp>
#include <ff/ubuffer.hpp>
#include <atomic>
#include <iostream>
#include <future>
#include <memory>
#include <thread>

using namespace ff;

/**
 * Incapsula un modello Producer-Consumer interno con due thread e due code per
 * disaccoppiare l'esecuzione dei task sull'acceleratore dal flusso principale
 * della pipeline FastFlow.
 */
class ff_node_acc_t : public ff_node {
 public:
   /**
    * @param acc Puntatore univoco a un'implementazione dell'interfaccia
    * IAccelerator (CPU, GPU, o FPGA).
    * @param count_promise Una std::promise per comunicare in modo sicuro il
    * conteggio finale dei task al thread main.
    */
   explicit ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                          std::promise<size_t> &&count_promise);
   ~ff_node_acc_t() override;

   long long getComputeTime_us() const;

   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   // Valore non nullo usato per segnalare la terminazione ai thread interni
   // attraverso le code.
   static void *const SENTINEL;

   void producerLoop();
   void consumerLoop();

   // Puntatore all'acceleratore concreto (CPU, GPU, FPGA).
   // std::unique_ptr garantisce la gestione automatica della memoria.
   std::unique_ptr<IAccelerator> accelerator_;

   // Tipi per le code interne Single-Producer/Single-Consumer.
   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;
   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   // Contatore atomico per il tempo di calcolo totale.
   std::atomic<long long> computed_us_{0};
   // Membri dedicati alla verifica finale del conteggio dei task.
   std::atomic<size_t> tasks_processed_{0};
   std::promise<size_t> count_promise_;

   // Oggetti thread che eseguono producerLoop e consumerLoop.
   std::thread prodTh_, consTh_;
};
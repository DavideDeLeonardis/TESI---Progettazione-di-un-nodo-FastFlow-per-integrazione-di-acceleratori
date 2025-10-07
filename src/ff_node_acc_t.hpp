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

class ff_node_acc_t : public ff_node {
 public:
   // Il costruttore accetta l'acceleratore e la "promise" per la verifica
   // finale
   explicit ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                          std::promise<size_t> &&count_promise);
   ~ff_node_acc_t() override;

   long long getComputeTime_us() const;

   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   static void *const SENTINEL;
   void producerLoop();
   void consumerLoop();

   // Membri per la gestione dell'acceleratore e delle code interne
   std::unique_ptr<IAccelerator> accelerator_;
   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;
   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};
   std::atomic<long long> computed_us_{0};

   // Thread interni per il disaccoppiamento
   std::thread prodTh_, consTh_;

   // Membri per la verifica del conteggio
   std::atomic<size_t> tasks_processed_;
   std::promise<size_t> count_promise_;
};
#pragma once
#include "../include/types.hpp"
#include "fastflow_includes.hpp"
#include <atomic>
#include <iostream>
#include <thread>

/* Nodo accelerato con doppia coda e due thread */
class ff_node_acc_t : public ff::ff_node {
 public:
   ff_node_acc_t() {
      std::cerr << "[ctor accNode]\n";
      std::cerr.flush();
   }
   ~ff_node_acc_t() {
      std::cerr << "[dtor accNode]\n";
      std::cerr.flush();
   }

   // FastFlow hooks
   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   void producerLoop();
   void consumerLoop();

   // bounded single-writer/single-reader queue (2 slot: task + sentinel)
   using TaskQ = ff::SWSR_Ptr_Buffer;
   using ResultQ = ff::SWSR_Ptr_Buffer;

   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   std::thread prodTh_, consTh_;
   std::atomic<bool> done_{false};
};

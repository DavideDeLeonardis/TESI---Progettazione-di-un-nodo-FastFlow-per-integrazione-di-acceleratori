#pragma once
#include "../include/types.hpp"
#include "fastflow_includes.hpp"
#include <atomic>
#include <iostream>
#include <thread>

/* Nodo accelerato con doppia coda e due thread              *
 *  - inQ_  : Task*  dalla porta FastFlow → producerLoop()    *
 *  - outQ_ : Result* da producerLoop()  → consumerLoop()    */
class ff_node_acc_t : public ff::ff_node {
 public:
   ff_node_acc_t() {
      std::cerr << "[ctor]\n";
      std::cerr.flush();
   }
   ~ff_node_acc_t() {
      std::cerr << "[dtor]\n";
      std::cerr.flush();
   }

   int svc_init() override;        // alloca code, avvia i thread
   void *svc(void *task) override; // riceve dallo stream FastFlow
   void svc_end() override;        // join thread, free code

 private:
   void producerLoop(); // pop inQ_  → compute → push outQ_
   void consumerLoop(); // pop outQ_ → ff_send_out()

   // bounded single-writer/single-reader pointer queue
   using TaskQ = ff::SWSR_Ptr_Buffer;
   using ResultQ = ff::SWSR_Ptr_Buffer;

   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   std::thread prodTh_, consTh_;
   std::atomic<bool> done_{false};
};

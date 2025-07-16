#pragma once
#include "../include/types.hpp"
#include "fastflow_includes.hpp"
#include <iostream>
#include <thread>

/* Nodo accelerato con doppia coda e due thread */
class ff_node_acc_t : public ff::ff_node {
 public:
   ff_node_acc_t() = default;
   ~ff_node_acc_t() {
      delete inQ_;
      delete outQ_;
   }

   int svc_init() override;     // init code + lancia producer+consumer
   void *svc(void *t) override; // riceve Task* o EOS
   void svc_end() override;     // (lasciamo vuoto)

 private:
   void producerLoop();
   void consumerLoop();

   using TaskQ = ff::uSWSR_Ptr_Buffer;
   using ResultQ = ff::uSWSR_Ptr_Buffer;

   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   std::thread prodTh_, consTh_;
};

#pragma once
#include "../include/types.hpp"
#include "fastflow_includes.hpp"
#include <atomic>
#include <iostream>
#include <thread>

/* Nodo accelerato con doppia coda e due thread */
class ff_node_acc_t : public ff_node {
 public:
   ff_node_acc_t();
   ~ff_node_acc_t();

   long long getComputeTime_us() const;

   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   static void *const SENTINEL;

   void producerLoop();
   void consumerLoop();

   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;

   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   std::thread prodTh_, consTh_;
   std::atomic<bool> done_{false};

   std::atomic<long long> computed_us_{0};

   std::atomic<size_t> inPushed_, inPopped_;
   std::atomic<size_t> outPushed_, outPopped_;
};

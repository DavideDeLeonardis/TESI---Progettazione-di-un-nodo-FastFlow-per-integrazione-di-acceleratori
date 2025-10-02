// File: src/ff_node_acc_t.hpp

#pragma once

#include "../include/types.hpp"
#include "IAccelerator.hpp" // <--- RIGA FONDAMENTALE! Assicurati che ci sia.
#include "fastflow_includes.hpp"
#include <atomic>
#include <iostream>
#include <memory> // <--- Assicurati che ci sia anche questa.
#include <thread>

class ff_node_acc_t : public ff_node {
 public:
   explicit ff_node_acc_t(std::unique_ptr<IAccelerator> acc);
   ~ff_node_acc_t() override;

   long long getComputeTime_us() const;

   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   static void *const SENTINEL;
   void producerLoop();
   void consumerLoop();

   std::unique_ptr<IAccelerator> accelerator_;

   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;
   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};
   std::thread prodTh_, consTh_;
   std::atomic<long long> computed_us_{0};
   std::atomic<size_t> inPushed_, inPopped_;
   std::atomic<size_t> outPushed_, outPopped_;
};
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

using namespace ff;

class ff_node_acc_t : public ff_node {
 public:
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
   std::unique_ptr<IAccelerator> accelerator_;
   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;
   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};
   std::atomic<long long> computed_us_{0};
   std::thread prodTh_, consTh_;
   std::atomic<size_t> tasks_processed_;
   std::promise<size_t> count_promise_;
};

#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <thread>

static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

ff_node_acc_t::ff_node_acc_t(std::unique_ptr<IAccelerator> acc)
    : accelerator_(std::move(acc)), computed_us_(0) {}

ff_node_acc_t::~ff_node_acc_t() {}

long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

int ff_node_acc_t::svc_init() {
   std::cerr << "[svc_init] Initializing accelerator...\n";
   if (!accelerator_ || !accelerator_->initialize()) {
      std::cerr << "Error: Accelerator initialization failed\n";
      return -1;
   }
   std::cerr << "[svc_init] Accelerator initialized successfully.\n";
   inQ_ = new TaskQ(1024);
   outQ_ = new ResultQ(1024);
   if (!inQ_->init() || !outQ_->init()) {
      std::cerr << "[svc_init] ERROR: init queues failed\n";
      return -1;
   }
   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);
   std::cerr << "[svc_init] Threads started\n";
   return 0;
}

void *ff_node_acc_t::svc(void *t) {
   std::cerr << "LOG 3: accNode::svc_run() received something\n";
   if (t == FF_EOS) {
      std::cerr << "LOG 98: accNode received EOS, pushing SENTINEL to inQ\n";
      while (!inQ_->push(SENTINEL))
         std::this_thread::yield();
      return FF_GO_ON;
   }
   std::cerr << "LOG 4: accNode pushing task to inQ\n";
   auto *task = static_cast<Task *>(t);
   while (!inQ_->push(task)) {
      std::this_thread::yield();
   }
   return FF_GO_ON;
}

void ff_node_acc_t::producerLoop() {
   std::cerr << "LOG P-A: producerLoop thread started and waiting\n";
   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr)) {
         std::this_thread::yield();
      }
      std::cerr << "LOG 5: Producer popped something from inQ\n";

      if (ptr == SENTINEL) {
         std::cerr << "LOG P-END: Producer received SENTINEL, forwarding and "
                      "exiting\n";
         while (!outQ_->push(SENTINEL)) {
            std::this_thread::yield();
         }
         break;
      }
      auto *task = static_cast<Task *>(ptr);
      std::cerr << "LOG P-B: Producer calling accelerator->execute()\n";
      long long current_task_us = 0;
      accelerator_->execute(task, current_task_us);
      computed_us_ += current_task_us;
      std::cerr << "LOG P-C: Producer returned from accelerator->execute()\n";

      auto *res = new Result{task->c, task->n};
      delete task; // Pulizia memoria del Task

      while (!outQ_->push(res)) {
         std::this_thread::yield();
      }
      std::cerr << "LOG 6: Producer pushed result to outQ\n";
   }
}

void ff_node_acc_t::consumerLoop() {
   std::cerr << "LOG C-A: consumerLoop thread started and waiting\n";
   void *ptr = nullptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "LOG C-B: Consumer popped something from outQ\n";

      if (ptr == SENTINEL) {
         std::cerr << "LOG C-END: Consumer received SENTINEL, sending EOS and "
                      "exiting\n";
         ff_send_out(FF_EOS);
         break;
      }
      std::cerr << "LOG C-C: Consumer calling ff_send_out() with a result\n";
      ff_send_out(ptr);
   }
}

void ff_node_acc_t::svc_end() {
   std::cerr
      << "[svc_end] Pipeline is shutting down. Signalling internal threads.\n";
   inQ_->push(SENTINEL);
   if (prodTh_.joinable())
      prodTh_.join();
   if (consTh_.joinable())
      consTh_.join();
   delete inQ_;
   delete outQ_;
   inQ_ = nullptr;
   outQ_ = nullptr;
}
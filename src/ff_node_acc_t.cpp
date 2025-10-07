#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <thread>

static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

ff_node_acc_t::ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                             std::promise<size_t> &&count_promise)
    : accelerator_(std::move(acc)), tasks_processed_(0),
      count_promise_(std::move(count_promise)), computed_us_(0) {}

ff_node_acc_t::~ff_node_acc_t() {
   std::cerr << "LOG D-A: Destructor called. Deleting queues...\n";
   delete inQ_;
   delete outQ_;
   std::cerr << "LOG D-B: Destructor finished.\n";
}

long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

int ff_node_acc_t::svc_init() {
   std::cerr << "LOG INIT-A: svc_init called.\n";
   if (!accelerator_ || !accelerator_->initialize()) {
      std::cerr << "Error: Accelerator initialization failed\n";
      return -1;
   }
   std::cerr << "LOG INIT-B: Accelerator initialized.\n";
   inQ_ = new TaskQ(1024);
   outQ_ = new ResultQ(1024);
   if (!inQ_->init() || !outQ_->init()) {
      std::cerr << "[svc_init] ERROR: init queues failed\n";
      return -1;
   }
   std::cerr << "LOG INIT-C: Queues created.\n";
   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);
   std::cerr << "LOG INIT-D: Threads started.\n";
   return 0;
}

void *ff_node_acc_t::svc(void *t) {
   std::cerr << "LOG S-1: accNode::svc() received something\n";
   if (t == FF_EOS) {
      std::cerr << "LOG S-2: accNode received EOS, pushing SENTINEL\n";
      if (inQ_)
         inQ_->push(SENTINEL);
      return FF_GO_ON;
   }
   std::cerr << "LOG S-3: accNode pushing task to inQ\n";
   auto *task = static_cast<Task *>(t);
   while (!inQ_->push(task))
      std::this_thread::yield();
   return FF_GO_ON;
}

void ff_node_acc_t::producerLoop() {
   std::cerr << "LOG P-A: producerLoop started and waiting.\n";
   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "LOG P-B: Producer popped something from inQ.\n";
      if (ptr == SENTINEL) {
         std::cerr << "LOG P-END: Producer received SENTINEL, forwarding.\n";
         while (!outQ_->push(SENTINEL))
            std::this_thread::yield();
         break;
      }
      auto *task = static_cast<Task *>(ptr);
      std::cerr << "LOG P-C: Producer calling accelerator->execute().\n";
      long long current_task_us = 0;
      accelerator_->execute(task, current_task_us);
      computed_us_ += current_task_us;
      std::cerr << "LOG P-D: Producer returned from accelerator->execute().\n";
      auto *res = new Result{task->c, task->n};
      delete task;
      while (!outQ_->push(res))
         std::this_thread::yield();
      std::cerr << "LOG P-E: Producer pushed result to outQ.\n";
   }
   std::cerr << "LOG P-EXIT: producerLoop exited.\n";
}

void ff_node_acc_t::consumerLoop() {
   std::cerr << "LOG K-A: consumerLoop started and waiting.\n";
   void *ptr = nullptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "LOG K-B: Consumer popped something from outQ.\n";
      if (ptr == SENTINEL) {
         std::cerr
            << "LOG K-END: Consumer received SENTINEL, fulfilling promise.\n";
         count_promise_.set_value(tasks_processed_.load());
         break;
      }
      std::cerr
         << "LOG K-C: Consumer processing result, incrementing counter.\n";
      tasks_processed_++;
      auto *res = static_cast<Result *>(ptr);
      delete res;
   }
   std::cerr << "LOG K-EXIT: consumerLoop exited.\n";
}

void ff_node_acc_t::svc_end() {
   std::cerr << "LOG END-A: svc_end called. Signalling threads.\n";
   if (inQ_) {
      inQ_->push(SENTINEL);
   }
   std::cerr << "LOG END-B: Joining threads...\n";
   if (prodTh_.joinable()) {
      prodTh_.join();
   }
   if (consTh_.joinable()) {
      consTh_.join();
   }
   std::cerr << "LOG END-C: Threads joined. svc_end finished.\n";
}
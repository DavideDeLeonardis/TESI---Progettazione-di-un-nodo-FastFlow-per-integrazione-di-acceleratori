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
   delete inQ_;
   delete outQ_;
}

long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

int ff_node_acc_t::svc_init() {
   std::cerr << "[INFO] Accelerator Node: Initializing...\n";
   if (!accelerator_ || !accelerator_->initialize()) {
      return -1;
   }
   inQ_ = new TaskQ(1024);
   outQ_ = new ResultQ(1024);
   if (!inQ_->init() || !outQ_->init()) {
      std::cerr << "[ERROR] Accelerator Node: Queues initialization failed.\n";
      return -1;
   }
   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);
   std::cerr << "[INFO] Accelerator Node: Internal threads started.\n";
   return 0;
}

void *ff_node_acc_t::svc(void *t) {
   if (t == FF_EOS) {
      if (inQ_)
         inQ_->push(SENTINEL);
      return FF_GO_ON;
   }
   auto *task = static_cast<Task *>(t);
   while (!inQ_->push(task))
      std::this_thread::yield();
   return FF_GO_ON;
}

void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();
      if (ptr == SENTINEL) {
         while (!outQ_->push(SENTINEL))
            std::this_thread::yield();
         break;
      }
      auto *task = static_cast<Task *>(ptr);
      long long current_task_us = 0;
      accelerator_->execute(task, current_task_us);
      computed_us_ += current_task_us;
      auto *res = new Result{task->c, task->n};
      delete task;
      while (!outQ_->push(res))
         std::this_thread::yield();
   }
}

void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      if (ptr == SENTINEL) {
         // Fine del flusso, comunica il conteggio finale al main thread
         count_promise_.set_value(tasks_processed_.load());
         break;
      }
      tasks_processed_++;
      auto *res = static_cast<Result *>(ptr);
      delete res; // Libera la memoria del Result
   }
}

void ff_node_acc_t::svc_end() {
   if (inQ_) {
      inQ_->push(SENTINEL);
   }
   if (prodTh_.joinable()) {
      prodTh_.join();
   }
   if (consTh_.joinable()) {
      consTh_.join();
   }
   std::cerr << "[INFO] Accelerator Node: Shutdown complete.\n";
}
// File: src/ff_node_acc_t.cpp

#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <thread>

static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

// COSTRUTTORE
ff_node_acc_t::ff_node_acc_t(std::unique_ptr<IAccelerator> acc)
    : accelerator_(std::move(acc)), inPushed_(0), inPopped_(0), outPushed_(0),
      outPopped_(0) {}

// DISTRUTTORE
ff_node_acc_t::~ff_node_acc_t() {}

// GETTER
long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

// SVC_INIT
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

// SVC
void *ff_node_acc_t::svc(void *t) {
   if (t == FF_EOS) {
      std::cerr << "[svc] received FF_EOS → pushing SENTINEL to inQ_\n";
      while (!inQ_->push(SENTINEL))
         std::this_thread::yield();
      return FF_GO_ON;
   }
   auto *task = static_cast<Task *>(t);
   while (!inQ_->push(task)) {
      std::this_thread::yield();
   }
   return FF_GO_ON;
}

// PRODUCER LOOP
void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr)) {
         std::this_thread::yield();
      }

      if (ptr == SENTINEL) {
         while (!outQ_->push(SENTINEL)) {
            std::this_thread::yield();
         }
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      long long current_task_us = 0;
      accelerator_->execute(task, current_task_us);
      computed_us_ += current_task_us;

      auto *res = new Result{task->c, task->n};
      while (!outQ_->push(res)) {
         std::this_thread::yield();
      }
   }
}

// CONSUMER LOOP
void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();

      if (ptr == SENTINEL) {
         ff_send_out(FF_EOS);
         break;
      }
      ff_send_out(ptr);
   }
}

// SVC_END
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
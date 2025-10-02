// File: ff_node_acc_t.cpp

#include "ff_node_acc_t.hpp"
#include <algorithm>
#include <iostream>
#include <thread>

// --- DEFINIZIONE DELLA SENTINELLA ---
// Inizializziamo il nostro puntatore SENTINEL.
// Creiamo un oggetto statico fittizio e usiamo il suo indirizzo.
// Questo garantisce che SENTINEL abbia un valore unico e non nullo.
static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;
// ------------------------------------

ff_node_acc_t::ff_node_acc_t()
    : inPushed_(0), inPopped_(0), outPushed_(0), outPopped_(0) {}

ff_node_acc_t::~ff_node_acc_t() {}

long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

int ff_node_acc_t::svc_init() {
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
   if (t == FF_EOS) {
      std::cerr << "[svc] received FF_EOS → pushing SENTINEL to inQ_\n";
      while (!inQ_->push(SENTINEL)) // Usa SENTINEL
         std::this_thread::yield();
      return FF_GO_ON;
   }

   auto *task = static_cast<Task *>(t);
   std::cerr << "[svc] received TASK ptr=" << task << " n=" << task->n << "\n";
   while (!inQ_->push(task)) {
      std::cerr << "[svc] inQ_ full, retrying\n";
      std::this_thread::yield();
   }
   std::cerr << "[svc] task pushed to inQ_, returning FF_GO_ON\n";
   return FF_GO_ON;
}

void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;
   while (true) {
      std::cerr << "[producer] waiting for inQ_ pop...\n";
      while (!inQ_->pop(&ptr)) {
         std::this_thread::yield();
      }
      size_t pc = ++inPopped_;
      std::cerr << "[producer] popped ptr (inPopped=" << pc << ")\n";

      if (ptr == SENTINEL) { // Controlla SENTINEL
         std::cerr
            << "[producer] got SENTINEL → forwarding SENTINEL to outQ_\n";
         while (!outQ_->push(SENTINEL)) { // Inoltra SENTINEL
            std::cerr << "[producer] outQ_ full pushing SENTINEL, retrying\n";
            std::this_thread::yield();
         }
         size_t op = ++outPushed_;
         std::cerr << "[producer] SENTINEL pushed to outQ_ (outPushed=" << op
                   << "), exiting producerLoop\n";
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      std::cerr << "[producer] computing task n=" << task->n << "\n";

      // --- INIZIO MISURAZIONE ---
      auto t0 = std::chrono::steady_clock::now();

      std::transform(task->a, task->a + task->n, task->b, task->c,
                     [](int x, int y) { return x + y; });

      auto t1 = std::chrono::steady_clock::now();
      auto us =
         std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      // Aggiungiamo il tempo misurato al totale (operazione atomica)
      computed_us_ += us;
      // --- FINE MISURAZIONE ---

      auto *res = new Result{task->c, task->n};
      std::cerr << "[producer] result ready\n";

      while (!outQ_->push(res)) {
         std::cerr << "[producer] outQ_ full, retrying push res\n";
         std::this_thread::yield();
      }
      size_t op2 = ++outPushed_;
      std::cerr << "[producer] result pushed to outQ_ (outPushed=" << op2
                << ")\n";
   }
   std::cerr << "[producer] exiting\n";
}

void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;
   while (true) {
      std::cerr << "[consumer] waiting for outQ_ pop...\n";
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      size_t opc = ++outPopped_;
      std::cerr << "[consumer] popped ptr (outPopped=" << opc << ")\n";

      if (ptr == SENTINEL) { // Controlla SENTINEL
         std::cerr << "[consumer] got SENTINEL → sending FF_EOS downstream\n";
         ff_send_out(FF_EOS);
         break;
      }

      std::cerr << "[consumer] sending result downstream\n";
      ff_send_out(ptr);
   }
   std::cerr << "[consumer] exiting\n";
}

void ff_node_acc_t::svc_end() {
   std::cerr
      << "[svc_end] Pipeline is shutting down. Signalling internal threads.\n";

   inQ_->push(SENTINEL); // Usa SENTINEL

   std::cerr << "[svc_end] joining producer thread\n";
   if (prodTh_.joinable())
      prodTh_.join();
   std::cerr << "[svc_end] producer thread joined\n";

   std::cerr << "[svc_end] joining consumer thread\n";
   if (consTh_.joinable())
      consTh_.join();
   std::cerr << "[svc_end] consumer thread joined\n";

   std::cerr << "[svc_end] Cleaning up resources.\n";
   delete inQ_;
   delete outQ_;
   inQ_ = nullptr;
   outQ_ = nullptr;
}
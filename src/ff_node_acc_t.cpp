#include "ff_node_acc_t.hpp"
#include <algorithm>
#include <iostream>
#include <thread>

// 1) svc_init: alloco le queue e lancio producer+consumer
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

// 2) svc: riceve Task* o EOS, li mette in inQ_, e ritorna GO_ON o EOS
void *ff_node_acc_t::svc(void *t) {
   if (t == EOS) {
      std::cerr << "[svc] received EOS → pushing sentinel(nullptr) to inQ_\n";
      while (!inQ_->push(nullptr)) {
         std::cerr << "[svc] retry EOS push to inQ_\n";
         std::this_thread::yield();
      }
      std::cerr << "[svc] EOS pushed, returning EOS\n";
      return EOS; // <-- qui segnaliamo la fine dello stream
   }
   // task “normale”
   auto *task = static_cast<Task *>(t);
   std::cerr << "[svc] received TASK ptr=" << task << " n=" << task->n
             << " → pushing to inQ_\n";
   while (!inQ_->push(task))
      std::this_thread::yield();
   return GO_ON;
}

// 3a) producerLoop: pop da inQ_, elabora, push in outQ_
void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;
   while (true) {
      std::cerr << "[producer] waiting for inQ_ pop...\n";
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[producer] popped ptr=" << ptr << "\n";

      if (ptr == nullptr) { // sentinel === nullptr
         std::cerr
            << "[producer] got sentinel → forwarding sentinel to outQ_\n";
         while (!outQ_->push(nullptr)) {
            std::cerr << "[producer] outQ_ full pushing sentinel, retrying\n";
            std::this_thread::yield();
         }
         std::cerr
            << "[producer] sentinel pushed to outQ_, exiting producerLoop\n";
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      std::cerr << "[producer] computing task ptr=" << task << " n=" << task->n
                << "\n";

      std::transform(task->a, task->a + task->n, task->b, task->c,
                     [](int x, int y) { return x + y; });

      auto *res = new Result{task->c, task->n};
      std::cerr << "[producer] result ready ptr=" << res
                << " → pushing to outQ_\n";

      while (!outQ_->push(res)) {
         std::cerr << "[producer] outQ_ full, retrying push res ptr=" << res
                   << "\n";
         std::this_thread::yield();
      }
      std::cerr << "[producer] result ptr=" << res << " pushed to outQ_\n";
   }
   std::cerr << "[producer] exiting\n";
}

// 3b) consumerLoop: pop da outQ_, invia tramite ff_send_out, e su sentinel
// manda EOS
void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;
   while (true) {
      std::cerr << "[consumer] waiting for outQ_ pop...\n";
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[consumer] popped ptr=" << ptr << "\n";

      if (ptr == nullptr) {
         std::cerr << "[consumer] got sentinel → sending EOS downstream\n";
         ff_send_out(EOS);
         break;
      }

      std::cerr << "[consumer] sending result ptr=" << ptr << "\n";
      ff_send_out(ptr);
   }
   std::cerr << "[consumer] exiting\n";
}

// 4) svc_end: attende la fine di producer + consumer
void ff_node_acc_t::svc_end() {

   std::cerr << "[svc_end] unblocking internal threads with sentinel\n";
   // Sblocca producerLoop
   while (!inQ_->push(nullptr))
      std::this_thread::yield();
   // Sblocca consumerLoop
   while (!outQ_->push(nullptr))
      std::this_thread::yield();

   std::cerr << "[svc_end] joining internal threads\n";

   if (prodTh_.joinable())
      prodTh_.join();

   if (consTh_.joinable())
      consTh_.join();

   std::cerr << "[svc_end] all threads terminated\n";
}

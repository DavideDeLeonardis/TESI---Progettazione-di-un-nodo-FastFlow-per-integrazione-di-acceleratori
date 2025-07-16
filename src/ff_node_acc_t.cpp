#include "ff_node_acc_t.hpp"
#include <algorithm>

// 1) svc_init: alloca code (size=2) e lancia i thread
int ff_node_acc_t::svc_init() {
   inQ_ = new TaskQ(2);
   outQ_ = new ResultQ(2);

   std::cerr << "[svc_init] inQ_=" << inQ_ << " outQ_=" << outQ_
             << " sizeof(*inQ_)=" << sizeof(*inQ_) << "\n";
   std::cerr.flush();

   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);

   std::cerr << "[svc_init] threads started, node @" << this << "\n";
   std::cerr.flush();
   return 0;
}

// 2) svc: riceve Task* o EOS, li accoda e segnala fine al filtro solo su EOS
void *ff_node_acc_t::svc(void *t) {
   std::cerr << "[svc ENTRY] ptr="
             << (t == EOS ? "EOS" : std::to_string((uintptr_t)t)) << std::endl;
   std::cerr.flush();

   if (t == EOS) {
      // Arriva il sentinel dal pipeline: sveglio il producer interno
      std::cerr << "[svc EOS] pushing nullptr sentinel\n";
      std::cerr.flush();
      inQ_->push(nullptr);
      // **MAI** restituire EOS qui: rimango vivo per far girare i thread
   } else {
      // È un Task: lo metto in coda
      Task *task = static_cast<Task *>(t);
      inQ_->push(task);
      std::cerr << "[svc TASK] queued ptr=" << task << std::endl;
      std::cerr.flush();
   }

   // In tutti i casi, ritorno GO_ON
   return GO_ON;
}

// 3) producerLoop: pop da inQ_, elabora, push in outQ_
void ff_node_acc_t::producerLoop() {
   std::cerr << "[producer@" << this << "] thread start\n";
   std::cerr.flush();

   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[producer] pop ptr=" << ptr << "\n";
      std::cerr.flush();
      if (ptr == nullptr) {
         // sentinel interno
         while (!outQ_->push(nullptr))
            std::this_thread::yield();
         break;
      }
      // compute
      auto *task = static_cast<Task *>(ptr);
      std::transform(task->a, task->a + task->n, task->b, task->c,
                     [](int x, int y) { return x + y; });
      // push risultato
      auto *res = new Result{task->c, task->n};
      while (!outQ_->push(res))
         std::this_thread::yield();
   }
   std::cerr << "[producer] EXIT\n";
   std::cerr.flush();
}

// 4) consumerLoop: pop da outQ_, ff_send_out() risultati e infine EOS
void ff_node_acc_t::consumerLoop() {
   void *ptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[consumer] pop ptr=" << ptr << std::endl;
      std::cerr.flush();
      if (ptr == nullptr) {
         // È il nostro sentinel interno → propago EOS al Collector
         ff_send_out(EOS);
         break;
      }
      // Altrimenti è un risultato valido
      ff_send_out(ptr);
   }
}

// 5) svc_end: segnala done, sblocca eventuali pop/push e join-a i thread
void ff_node_acc_t::svc_end() {
   done_ = true;
   inQ_->push(nullptr);
   outQ_->push(nullptr);
   if (prodTh_.joinable())
      prodTh_.join();
   if (consTh_.joinable())
      consTh_.join();
}

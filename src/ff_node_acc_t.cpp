#include "ff_node_acc_t.hpp"

// 1) svc_init: alloca le queue e parte coi thread
int ff_node_acc_t::svc_init() {
   // due slot bastano: uno per il Task, uno per il sentinel
   inQ_ = new TaskQ(2);
   outQ_ = new ResultQ(2);

   std::cerr << "[svc_init] inQ_=" << inQ_ << "  outQ_=" << outQ_
             << "  sizeof(*inQ_)=" << sizeof(*inQ_) << std::endl;
   std::cerr.flush();

   // lancio subito i due thread di producer/consumer
   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);

   std::cerr << "[svc_init] threads started, node @" << this << std::endl;
   std::cerr.flush();
   return 0;
}

// 2) svc: riceve Task* o EOS dal pipeline, li mette in inQ_
void *ff_node_acc_t::svc(void *t) {
   std::cerr << "[svc ENTRY] ptr="
             << (t == EOS ? "EOS" : std::to_string((uintptr_t)t)) << std::endl;
   std::cerr.flush();

   if (t == EOS) {
      // arriva il “fine‐stream” dall’Emitter: sveglio producerLoop e termino
      std::cerr << "[svc EOS] pushing nullptr sentinel\n";
      std::cerr.flush();
      inQ_->push(nullptr);
      return EOS;
   }
   // è un Task normale: lo accodo e rimango vivo
   {
      Task *task = static_cast<Task *>(t);
      bool ok = inQ_->push(task);
      std::cerr << "[svc TASK] push(task) ok=" << ok << " ptr=" << task
                << std::endl;
      std::cerr.flush();
      // non dovrebbe mai fallire con capacity=2
      if (!ok) {
         while (!inQ_->push(task))
            std::this_thread::yield();
      }
   }
   return GO_ON;
}

// 3) producerLoop: estrae da inQ_, calcola, inserisce in outQ_
void ff_node_acc_t::producerLoop() {
   std::cerr << "[producer@" << this << "] thread start" << std::endl;
   std::cerr.flush();

   void *ptr = nullptr;
   while (true) {
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[producer] pop ptr=" << ptr << std::endl;
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
      // push result
      auto *res = new Result{task->c, task->n};
      while (!outQ_->push(res))
         std::this_thread::yield();
   }
   std::cerr << "[producer] EXIT\n";
   std::cerr.flush();
}

// 4) consumerLoop: estrae da outQ_ e invia al Collector
void ff_node_acc_t::consumerLoop() {
   void *ptr;
   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      std::cerr << "[consumer] pop ptr=" << ptr << std::endl;
      std::cerr.flush();
      if (ptr == nullptr) {
         // inoltro l’EOS al Collector e esco
         ff_send_out(EOS);
         break;
      }
      // altrimenti risultato valido
      ff_send_out(ptr);
   }
}

// 5) svc_end: join dei thread
void ff_node_acc_t::svc_end() {
   done_ = true;
   inQ_->push(nullptr);
   outQ_->push(nullptr);
   if (prodTh_.joinable())
      prodTh_.join();
   if (consTh_.joinable())
      consTh_.join();
}

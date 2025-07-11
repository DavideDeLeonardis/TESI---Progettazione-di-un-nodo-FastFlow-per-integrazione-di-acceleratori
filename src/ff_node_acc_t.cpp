#include "ff_node_acc_t.hpp"
#include <algorithm>
#include <chrono>
#include <ff/node.hpp>
#include <iostream>

void *ff_node_acc_t::svc(void *t) {
   if (t == EOS)
      return EOS;

   using clock = std::chrono::steady_clock;
   auto *task = static_cast<Task *>(t);

   const auto c0 = clock::now();
   std::transform(task->a, task->a + task->n, task->b, task->c,
                  [](int x, int y) { return x + y; });
   const auto c1 = clock::now();

   using us = std::chrono::microseconds;
   const auto compute_us = std::chrono::duration_cast<us>(c1 - c0).count();

   std::cout << "compute_us=" << compute_us << " µs\n";

   return new Result{task->c, task->n}; // al Collector
}

// /* ---------- svc_init ---------- */
// int ff_node_acc_t::svc_init() {
//    std::cerr << "[init] starting\n";
//    inQ_ = new ff::SWSR_Ptr_Buffer(64);
//    assert(inQ_ && "[init] queue alloc failed");
//    prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
//    return 0; // deve restituire 0
// }

// /* ---------- svc ---------- */
// void *ff_node_acc_t::svc(void *t) {
//    if (t == EOS) {         // EOS dall’Emitter
//       inQ_->push(nullptr); // sentinel
//       return GO_ON;        // non chiudiamo qui
//    }
//    inQ_->push(t);
//    std::cerr << "[svc] got EOS, pushed sentinel\n";
//    return GO_ON;
// }

// /* ---------- producerLoop ---------- */
// void ff_node_acc_t::producerLoop() {
//    void *p;
//    while (true) {
//       inQ_->pop(&p);
//       if (!p) {                // sentinel
//          ff_send_out(nullptr); // UNICO EOS
//          break;
//       }
//       auto *task = static_cast<Task *>(p);
//       std::transform(task->a, task->a + task->n, task->b, task->c,
//                      [](int x, int y) { return x + y; });
//       std::cerr << "[producer] sending EOS\n";
//       ff_send_out(new Result{task->c, task->n});
//    }
// }

// /* ---------- svc_end ---------- */
// void ff_node_acc_t::svc_end() {
//    if (prodTh_.joinable())
//       prodTh_.join(); // ora il producer è già finito
//    delete inQ_;
// }

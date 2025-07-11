#include "ff_node_acc_t.hpp"
#include <algorithm> // std::transform
#include <iterator>  // std::begin / std::end

/* ----------- svc_init: nulla da fare per CPU baseline ------------ */
int ff_node_acc_t::svc_init() {
   return 0; // OK
}

/* --------------------- svc: core computazione -------------------- */
void *ff_node_acc_t::svc(void *t) {
   if (!t)
      return EOS; // sentinel -> fine stream

   auto *task = static_cast<Task *>(t);

   // CPU baseline: sommiamo A + B -> C
   std::transform(task->a, task->a + task->n, task->b, task->c,
                  [](int x, int y) { return x + y; });

   // Restituiamo un Result (potremmo riusare Task; qui ne creiamo uno stack)
   auto *res = new Result{task->c, task->n};
   return static_cast<void *>(res);
}

/* -------------------- svc_end: nulla da fare --------------------- */
void ff_node_acc_t::svc_end() {
   // nop
}

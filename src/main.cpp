#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include "CpuAccelerator.hpp"
#include "GpuAccelerator.hpp"
#include "FpgaAccelerator.hpp"

/* ------------ Emitter ------------- */
class Emitter : public ff::ff_node {
 public:
   // Il costruttore ora accetta anche il numero di task da inviare
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      a.resize(n);
      b.resize(n);
      c.resize(n);
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }
      // Prepariamo un singolo "stampo" per il task.
      // Invieremo il puntatore a questo stesso task più volte.
      // Funziona perché la pipeline li processa uno alla volta.
      task = {a.data(), b.data(), c.data(), n};
   }

   void *svc(void *) override {
      // Invia un task finché non abbiamo raggiunto il numero desiderato
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         return &task;
      }
      // Una volta finiti tutti i task, invia il segnale di fine stream
      return FF_EOS;
   }

   const std::vector<int> &getA() const { return a; }
   const std::vector<int> &getB() const { return b; }
   std::vector<int> &getC() { return c; }

 private:
   size_t tasks_to_send;
   size_t tasks_sent;
   Task task;
   std::vector<int> a, b, c;
};

/* ----------- Collector ----------- */
// La classe Collector non ha bisogno di alcuna modifica.
// Processerà i risultati man mano che arrivano.
class Collector : public ff::ff_node {
 public:
   Collector(const std::vector<int> &A, const std::vector<int> &B,
             std::vector<int> &C)
       : a(A), b(B), c(C) {}

   void *svc(void *r) override {
      if (r == FF_EOS) {
         return FF_EOS;
      }

      auto *res = static_cast<Result *>(r);
      bool ok = true;
      // La validazione rimane la stessa
      for (size_t i = 0; i < res->n; i += res->n / 16 + 1) {
         if (c[i] != a[i] + b[i]) {
            ok = false;
            std::cerr << "[collector] VALIDATION FAILED on task result!\n";
            break;
         }
      }
      if (ok) {
         std::cerr << "[collector] Task result OK\n";
      }

      delete res;
      return FF_GO_ON;
   }

 private:
   const std::vector<int> &a;
   const std::vector<int> &b;
   std::vector<int> &c;
};

/* --------------- main --------------- */
int main(int argc, char *argv[]) {
   // ARGS:
   // 1: Dimensione Vettore (N)          (default: 1'000'000)
   // 2: Numero di Task (NUM_TASKS)      (default: 100)
   // 3: Device ('cpu', 'gpu' o 'fpga')  (default: 'cpu')

   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
   size_t NUM_TASKS = (argc > 2 ? std::stoull(argv[2]) : 100);
   std::string device_type = (argc > 3 ? argv[3] : "cpu");

   std::cout << "Configuration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n";

   Emitter emitter(N, NUM_TASKS);

   std::unique_ptr<IAccelerator> accelerator;

   if (device_type == "fpga") {
      std::cout << "Using FPGA Accelerator...\n";
      accelerator = std::make_unique<FpgaAccelerator>();
   } else if (device_type == "gpu") {
      std::cout << "Using GPU Accelerator...\n";
      accelerator = std::make_unique<GpuAccelerator>();
   } else {
      std::cout << "Using CPU Accelerator...\n";
      accelerator = std::make_unique<CpuAccelerator>();
   }

   ff_node_acc_t accNode(std::move(accelerator));

   Collector collector(emitter.getA(), emitter.getB(), emitter.getC());

   ff::ff_Pipe<> pipe(false, &emitter, &accNode, &collector);

   auto t0 = std::chrono::steady_clock::now();
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "run error\n";
      return -1;
   }
   auto t1 = std::chrono::steady_clock::now();
   auto us_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   auto us_computed = accNode.getComputeTime_us();

   std::cout << "-------------------------------------------\n"
             << "Total time for " << NUM_TASKS << " tasks:\n"
             << "N=" << N << " elapsed=" << us_elapsed << " µs"
             << ", computed=" << us_computed << " µs\n"
             << "-------------------------------------------\n"
             << "Average time per task:\n"
             << "Avg elapsed=" << us_elapsed / NUM_TASKS << " µs/task\n"
             << "Avg computed=" << us_computed / NUM_TASKS << " µs/task\n"
             << "-------------------------------------------\n";

   return 0;
}
#include "CpuAccelerator.hpp"
#include "FpgaAccelerator.hpp"
#include "GpuAccelerator.hpp"
#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* ------------ Emitter ------------- */
class Emitter : public ff::ff_node {
 public:
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      a.resize(n);
      b.resize(n);
      c.resize(n);
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }
      a_ptr_ = a.data();
      b_ptr_ = b.data();
      c_ptr_ = c.data();
      n_ = n;
   }

   void *svc(void *) override {
      std::cerr << "LOG 1: Emitter::svc_run() called\n";
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         std::cerr << "LOG 2: Emitter producing new task " << tasks_sent
                   << "\n";
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_};
      }
      std::cerr << "LOG 99: Emitter sending EOS\n";
      return FF_EOS;
   }

   const std::vector<int> &getA() const { return a; }
   const std::vector<int> &getB() const { return b; }
   std::vector<int> &getC() { return c; }

 private:
   size_t tasks_to_send;
   size_t tasks_sent;
   int *a_ptr_, *b_ptr_, *c_ptr_;
   size_t n_;
   std::vector<int> a, b, c;
};

/* ----------- Collector ----------- */
class Collector : public ff::ff_node {
 public:
   Collector(const std::vector<int> &A, const std::vector<int> &B,
             std::vector<int> &C)
       : a(A), b(B), c(C) {}

   void *svc(void *r) override {
      std::cerr << "LOG 7: Collector::svc_run() received something\n";
      if (r == FF_EOS) {
         std::cerr << "LOG 100: Collector received EOS\n";
         return FF_EOS;
      }

      std::cerr << "LOG 8: Collector processing a result\n";
      auto *res = static_cast<Result *>(r);
      bool ok = true;
      for (size_t i = 0; i < res->n; i += res->n / 16 + 1) {
         if (c[i] != a[i] + b[i]) {
            ok = false;
            break;
         }
      }
      if (ok) {
         std::cerr << "LOG 8a: Collector VALIDATION OK\n";
      } else {
         std::cerr << "LOG 8b: Collector VALIDATION FAILED\n";
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
   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
   size_t NUM_TASKS = (argc > 2 ? std::stoull(argv[2]) : 100);
   std::string device_type = (argc > 3 ? argv[3] : "gpu");

   std::cout << "Configuration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n";

   Emitter emitter(N, NUM_TASKS);
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga") {
      accelerator = std::make_unique<FpgaAccelerator>();
   } else if (device_type == "gpu") {
      accelerator = std::make_unique<GpuAccelerator>();
   } else {
      accelerator = std::make_unique<CpuAccelerator>();
   }
   ff_node_acc_t accNode(std::move(accelerator));
   Collector collector(emitter.getA(), emitter.getB(), emitter.getC());
   ff::ff_Pipe<> pipe(false, &emitter, &accNode, &collector);
   auto t0 = std::chrono::steady_clock::now();
   pipe.run_and_wait_end();
   auto t1 = std::chrono::steady_clock::now();
   auto us_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
   auto us_computed = accNode.getComputeTime_us();

   std::cout << "-------------------------------------------\n"
             << "Total time for " << NUM_TASKS << " tasks on " << device_type
             << ":\n"
             << "N=" << N << " elapsed=" << us_elapsed << " µs"
             << ", computed=" << us_computed << " µs\n"
             << "-------------------------------------------\n"
             << "Average time per task:\n"
             << "Avg elapsed=" << us_elapsed / (NUM_TASKS == 0 ? 1 : NUM_TASKS)
             << " µs/task\n"
             << "Avg computed="
             << us_computed / (NUM_TASKS == 0 ? 1 : NUM_TASKS) << " µs/task\n"
             << "-------------------------------------------\n";
   return 0;
}
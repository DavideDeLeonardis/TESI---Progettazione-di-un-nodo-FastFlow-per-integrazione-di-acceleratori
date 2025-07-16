#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <vector>

/* ------------ Emitter ------------- */
class Emitter : public ff::ff_node {
 public:
   explicit Emitter(size_t n) : sent(false) {
      a.resize(n);
      b.resize(n);
      c.resize(n);
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }
      task = {a.data(), b.data(), c.data(), n};
   }
   void *svc(void *) override {
      if (!sent) {
         sent = true;
         return &task;
      }
      return EOS;
   }
   const std::vector<int> &getA() const { return a; }
   const std::vector<int> &getB() const { return b; }
   std::vector<int> &getC() { return c; }

 private:
   bool sent;
   Task task;
   std::vector<int> a, b, c;
};

/* ----------- Collector ----------- */
class Collector : public ff::ff_node {
 public:
   Collector(const std::vector<int> &A, const std::vector<int> &B,
             std::vector<int> &C)
       : a(A), b(B), c(C) {}

   void *svc(void *r) override {
      // doppio log su stderr
      std::cerr << "[collector] svc(r="
                << (r == EOS ? "EOS" : std::to_string((uintptr_t)r)) << ")\n";
      std::cerr.flush();

      if (r == EOS) {
         std::cerr << "[collector] received EOS, terminating\n";
         std::cerr.flush();
         return EOS; // chiude la pipeline
      }

      auto *res = static_cast<Result *>(r);
      bool ok = true;
      for (size_t i = 0; i < res->n; i += res->n / 16 + 1) {
         if (c[i] != a[i] + b[i]) {
            ok = false;
            break;
         }
      }

      // stampo su stderr per sicurezza
      std::cerr << "[collector] result is "
                << (ok ? "CPU baseline OK" : "Baseline FAILED") << "\n";

      delete res;
      return GO_ON;
   }

 private:
   const std::vector<int> &a;
   const std::vector<int> &b;
   std::vector<int> &c;
};

/* --------------- main --------------- */
int main(int argc, char *argv[]) {
   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);

   Emitter emitter(N);
   ff_node_acc_t accNode;
   Collector collector(emitter.getA(), emitter.getB(), emitter.getC());

   ff::ff_Pipe<> pipe(false, &emitter, &accNode, &collector);

   auto t0 = std::chrono::steady_clock::now();
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "run error\n";
      return -1;
   }
   auto t1 = std::chrono::steady_clock::now();
   auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
   std::cout << "N=" << N << " elapsed=" << us << " µs\n";
   return 0;
}

#include "CpuAccelerator.hpp"
#include "FpgaAccelerator.hpp"
#include "GpuAccelerator.hpp"
#include "ff_node_acc_t.hpp"
#include <chrono>
#include <future>
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
      std::cerr << "LOG E-1: Emitter::svc() called\n";
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         std::cerr << "LOG E-2: Emitter producing new task " << tasks_sent
                   << "\n";
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_};
      }
      std::cerr << "LOG E-99: Emitter sending EOS\n";
      return FF_EOS;
   }

 private:
   size_t tasks_to_send, tasks_sent;
   int *a_ptr_, *b_ptr_, *c_ptr_;
   size_t n_;
   std::vector<int> a, b, c;
};

/* --------------- main --------------- */
int main(int argc, char *argv[]) {
   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
   size_t NUM_TASKS = (argc > 2 ? std::stoull(argv[2]) : 100);
   std::string device_type = (argc > 3 ? argv[3] : "cpu");

   std::cout << "LOG M-1: Main started. Configuration: N=" << N
             << ", NUM_TASKS=" << NUM_TASKS << ", Device=" << device_type
             << "\n";
   Emitter emitter(N, NUM_TASKS);
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga")
      accelerator = std::make_unique<FpgaAccelerator>();
   else if (device_type == "gpu")
      accelerator = std::make_unique<GpuAccelerator>();
   else
      accelerator = std::make_unique<CpuAccelerator>();

   std::promise<size_t> count_promise;
   std::future<size_t> count_future = count_promise.get_future();

   ff_node_acc_t accNode(std::move(accelerator), std::move(count_promise));

   ff::ff_Pipe<> pipe(false, &emitter, &accNode);

   std::cerr << "LOG M-2: Calling pipe.run_and_wait_end()...\n";
   auto t0 = std::chrono::steady_clock::now();
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "run error\n";
      return -1;
   }
   auto t1 = std::chrono::steady_clock::now();
   std::cerr << "LOG M-3: pipe.run_and_wait_end() returned.\n";

   std::cerr << "LOG M-4: Calling count_future.get()...\n";
   size_t final_count = count_future.get();
   std::cerr << "LOG M-5: count_future.get() returned with count "
             << final_count << ".\n";

   auto us_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
   auto us_computed = accNode.getComputeTime_us();

   std::cout << "-------------------------------------------\n"
             << "Verification:\n"
             << "Tasks processed: " << final_count << " / " << NUM_TASKS
             << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
             << "-------------------------------------------\n";
   std::cerr << "LOG M-6: Main is about to exit.\n";
   return 0;
}
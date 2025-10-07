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
// Questo nodo produce i Task da inviare alla pipeline.
class Emitter : public ff::ff_node {
 public:
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      // Alloca i vettori una sola volta per efficienza
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
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         // Crea un nuovo oggetto Task sulla heap per ogni task
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_};
      }
      // Dopo aver inviato tutti i task, invia il segnale di fine stream
      return FF_EOS;
   }

 private:
   size_t tasks_to_send;
   size_t tasks_sent;
   int *a_ptr_, *b_ptr_, *c_ptr_;
   size_t n_;
   std::vector<int> a, b, c;
};

/* --------------- main --------------- */
int main(int argc, char *argv[]) {
   // Parsing degli argomenti da riga di comando con valori di default
   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
   size_t NUM_TASKS = (argc > 2 ? std::stoull(argv[2]) : 100);
   std::string device_type = (argc > 3 ? argv[3] : "cpu");

   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n";

   Emitter emitter(N, NUM_TASKS);

   // Selezione dell'acceleratore (CPU, GPU, o FPGA) in base all'argomento
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga")
      accelerator = std::make_unique<FpgaAccelerator>();
   else if (device_type == "gpu")
      accelerator = std::make_unique<GpuAccelerator>();
   else
      accelerator = std::make_unique<CpuAccelerator>();

   // Creazione della coppia promise/future per la verifica finale
   std::promise<size_t> count_promise;
   std::future<size_t> count_future = count_promise.get_future();

   // Creazione del nodo accelerato, passando l'acceleratore e la promise
   ff_node_acc_t accNode(std::move(accelerator), std::move(count_promise));

   // Creazione della pipeline a 2 stadi: Emitter -> accNode
   ff::ff_Pipe<> pipe(false, &emitter, &accNode);

   std::cout << "[Main] Starting pipeline execution...\n";
   auto t0 = std::chrono::steady_clock::now();
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "[ERROR] Main: Pipeline execution failed.\n";
      return -1;
   }
   auto t1 = std::chrono::steady_clock::now();
   std::cout << "[Main] Pipeline execution finished.\n";

   // Attesa sincrona del conteggio finale comunicato dal nodo accelerato
   size_t final_count = count_future.get();

   // Raccolta e stampa dei risultati di performance
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
             << "-------------------------------------------\n"
             << "Verification:\n"
             << "Tasks processed: " << final_count << " / " << NUM_TASKS
             << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
             << "-------------------------------------------\n";
   return 0;
}
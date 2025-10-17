#include "../../include/ff_includes.hpp"
#include "accelerator/FpgaAccelerator.hpp"
#include "accelerator/GpuAccelerator.hpp"
#include "accelerator/ff_node_acc_t.hpp"
#include "cpu_runner/CpuParallelRunner.hpp"
#include "helpers/Helpers.hpp"
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Nodo sorgente della pipeline FastFlow.
 *
 * Il nodo Emitter genera i Task da far processare al nodo ff_node_acc_t.
 * Inizializza i dati di input una sola volta, poi crea dinamicamente un nuovo
 * oggetto Task per ogni richiesta dalla pipeline.
 */
class Emitter : public ff_node {
 public:
   /**
    * @param n Dimensione dei vettori da processare.
    * @param num_tasks Il numero totale di task da generare.
    */
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      // Init dei vettori con i dati di input.
      a.resize(n);
      b.resize(n);
      c.resize(n);
      // Usiamo 2 vettori con dati diversi cosi un compilatore estremamente
      // intelligente non bara e non trasforma la somma in una moltiplicazione
      // (2 * a[i]).
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }

      a_ptr_ = a.data();
      b_ptr_ = b.data();
      c_ptr_ = c.data();
      n_ = n;
   }

   /**
    * @brief Genera un nuovo Task fino al raggiungimento del numero totale.
    * @return Un puntatore a un nuovo Task, o FF_EOS al termine.
    */
   void *svc(void *) override {
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_, tasks_sent};
      }

      // Una volta inviati tutti i task -> fine stream.
      return FF_EOS;
   }

 private:
   size_t tasks_to_send;          // Numero totale di task da inviare
   size_t tasks_sent;             // Numero di task già inviati
   std::vector<int> a, b, c;      // Vettori di input/output
   int *a_ptr_, *b_ptr_, *c_ptr_; // Puntatori ai dati di input/output
   size_t n_;                     // Dimensione dei vettori
};

/**
 * @brief Orchestra l'intera pipeline FastFlow per l'offloading su un
 * acceleratore. Crea i due nodi della pipeline FF (Emitter, ff_node_acc_t).
 * Riceve l'acceleratore già inizializzato. Avvia la pipeline. Misura e
 * raccoglie i tempi di esecuzione (computed ed elapsed) e il numero di task
 * completati.
 */
void runAcceleratorPipeline(size_t N, size_t NUM_TASKS,
                            IAccelerator *accelerator, long long &ns_elapsed,
                            long long &ns_computed, size_t &final_count) {

   // Dati per ottenere il conteggio finale dei task processati.
   StatsCollector stats;
   std::future<size_t> count_future = stats.count_promise.get_future();

   // Creazione della pipeline FF e dei suoi due nodi (Emitter, ff_node_acc_t),
   // il cui secondo nodo incapsula una pipeline interna a 2 thread (producer,
   // consumer).
   Emitter emitter(N, NUM_TASKS);
   ff_node_acc_t accNode(accelerator, &stats);
   ff_Pipe<> pipe(&emitter, &accNode);

   std::cout << "[Main] Starting FF pipeline execution...\n";
   auto t0 = std::chrono::steady_clock::now();

   // Avvio della pipeline e attesa del completamento.
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "[ERROR] Main: Pipeline execution failed.\n";
      ns_elapsed = 0;
      ns_computed = 0;
      final_count = 0;
      return;
   }
   auto t1 = std::chrono::steady_clock::now();
   std::cout << "[Main] FF Pipeline execution finished.\n";

   // Raccolta dei risultati.
   final_count = count_future.get();
   ns_elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
   ns_computed = stats.computed_ns.load();
}

int main(int argc, char *argv[]) {
   // Parametri della command line.
   size_t N = 1000000, NUM_TASKS = 50; // Default
   std::string device_type = "cpu";    // Default
   std::string kernel_path;
   std::string kernel_name;

   long long ns_elapsed = 0;  // Tempo totale (host) per completare tutti i task
   long long ns_computed = 0; // Tempo totale di calcolo (device)
   size_t final_count = 0;    // Numero totale di task effettivamente completati

   // Parsing degli argomenti della command line.
   parse_args(argc, argv, N, NUM_TASKS, device_type, kernel_path, kernel_name);

   print_configuration(N, NUM_TASKS, device_type, kernel_path);

   // In base al device scelto, esegue la parallelizzazione dei task su CPU
   // multicore tramite ff o la pipeline con offloading su GPU/FPGA.
   if (device_type == "cpu") {
      ns_elapsed = executeCpuParallelTasks(N, NUM_TASKS, final_count);

      // Su CPU non c'è overhead di trasferimento dati --> computed = elapsed.
      ns_computed = ns_elapsed;

   } else if (device_type == "gpu") {
      auto accelerator =
         std::make_unique<GpuAccelerator>(kernel_path, kernel_name);
      runAcceleratorPipeline(N, NUM_TASKS, accelerator.get(), ns_elapsed,
                             ns_computed, final_count);

   } else if (device_type == "fpga") {
      auto accelerator =
         std::make_unique<FpgaAccelerator>(kernel_path, kernel_name);
      runAcceleratorPipeline(N, NUM_TASKS, accelerator.get(), ns_elapsed,
                             ns_computed, final_count);

   } else {
      std::cerr << "[ERROR] Invalid device type '" << device_type << "'.\n\n";
      print_usage(argv[0]);
      return -1;
   }

   print_stats(N, NUM_TASKS, device_type, ns_elapsed, ns_computed, final_count);

   return 0;
}
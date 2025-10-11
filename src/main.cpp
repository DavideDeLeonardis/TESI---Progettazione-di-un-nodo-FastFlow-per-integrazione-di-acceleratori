#include "../include/ff_includes.hpp"
#include "CpuParallelRunner.hpp"
#include "FpgaAccelerator.hpp"
#include "GpuAccelerator.hpp"
#include "ff_node_acc_t.hpp"
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
 * Emitter genera i Task da processare.
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
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }

      // Salvataggio dei ptr ai dati di input/output e della dim dei vettori.
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
 * acceleratore. Crea il nodo sorgente (Emitter) che genera i task. Seleziona,
 * crea e inizializza l'oggetto acceleratore corretto. Imposta la pipeline
 * FastFlow a 2 stadi (Emitter -> ff_node_acc_t). Avvia la pipeline, misura
 * il tempo totale di esecuzione (elapsed). Raccoglie i risultati finali, come
 * il tempo di calcolo puro (computed) e il numero di task completati.
 */
void runAcceleratorPipeline(size_t N, size_t NUM_TASKS,
                            const std::string &device_type,
                            long long &ns_elapsed, long long &ns_computed,
                            size_t &final_count) {

   // Creazione del nodo sorgente della pipeline FF.
   Emitter emitter(N, NUM_TASKS);

   // Selezione e creazione dell'acceleratore.
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga")
      accelerator = std::make_unique<FpgaAccelerator>();
   else if (device_type == "gpu")
      accelerator = std::make_unique<GpuAccelerator>();

   // Dati per ottenere il conteggio finale dei task processati.
   StatsCollector stats;
   std::future<size_t> count_future = stats.count_promise.get_future();

   // Creazione del nodo di calcolo che usa l'acceleratore scelto.
   ff_node_acc_t accNode(std::move(accelerator), &stats);

   // Creazione della pipeline FF a 2 stadi, il cui secondo stadio incapsula
   // una pipeline interna a 2 thread (producer, consumer).
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
   // ------ Parsing degli argomenti della riga di comando ------
   size_t N = 1000000, NUM_TASKS = 50;
   std::string device_type = "cpu";

   if (argc > 1 &&
       (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
   }

   try {
      if (argc > 1)
         N = std::stoull(argv[1]);
      if (argc > 2)
         NUM_TASKS = std::stoull(argv[2]);
      if (argc > 3)
         device_type = argv[3];
   } catch (const std::invalid_argument &e) {
      std::cerr << "[ERROR] Invalid numeric argument provided.\n\n";
      print_usage(argv[0]);
      return -1;
   }

   if (N == 0) {
      std::cerr << "[FATAL] La dimensione dei vettori (N) non può essere 0.\n";
      return EXIT_FAILURE;
   }
   // /----- Parsing degli argomenti della riga di comando ------

   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n\n";

   long long ns_elapsed = 0;
   long long ns_computed = 0;
   size_t final_count = 0;

   // In base al device scelto, esegue la parallelizzazione dei task su CPU
   // multicore o la pipeline con offloading su GPU/FPGA.
   if (device_type == "cpu") {
      ns_elapsed = executeCpuParallelTasks(N, NUM_TASKS);

      // Su CPU non c'è overhead di trasferimento, quindi elapsed = computed.
      ns_computed = ns_elapsed;
      final_count = NUM_TASKS;

   } else if (device_type == "gpu" || device_type == "fpga")
      runAcceleratorPipeline(N, NUM_TASKS, device_type, ns_elapsed, ns_computed,
                             final_count);
   else {
      std::cerr << "[ERROR] Invalid device type '" << device_type << "'.\n\n";
      print_usage(argv[0]);
      return -1;
   }

   print_stats(N, NUM_TASKS, device_type, ns_elapsed, ns_computed, final_count);

   return 0;
}
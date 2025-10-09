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
/**
 * @brief Nodo sorgente della pipeline FastFlow.
 *
 * Emitter genera i Task da processare.
 * Inizializza i dati di input una sola volta e poi crea dinamicamente un nuovo
 * oggetto Task per ogni richiesta dalla pipeline.
 */
class Emitter : public ff::ff_node {
 public:
   /**
    * @param n Dimensione dei vettori da processare.
    * @param num_tasks Il numero totale di task da generare.
    */
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      a.resize(n);
      b.resize(n);
      c.resize(n);
      for (size_t i = 0; i < n; ++i) {
         a[i] = int(i);
         b[i] = int(2 * i);
      }

      // Salviamo i puntatori ai dati e la dimensione per creare i Task.
      a_ptr_ = a.data();
      b_ptr_ = b.data();
      c_ptr_ = c.data();
      n_ = n;
   }

   /**
    * @return Un puntatore a un nuovo Task, o FF_EOS al termine.
    */
   void *svc(void *) override {
      if (tasks_sent < tasks_to_send) {
         tasks_sent++;
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_};
      }

      // Una volta inviati tutti i task segnala il fine stream.
      return FF_EOS;
   }

 private:
   size_t tasks_to_send;          // Numero totale di task da inviare
   size_t tasks_sent;             // Numero di task già inviati
   int *a_ptr_, *b_ptr_, *c_ptr_; // Puntatori ai dati
   size_t n_;                     // Dimensione dei vettori
   std::vector<int> a, b, c;      // I vettori che possiedono i dati.
};

// Helper per stampare le istruzioni d'uso
void print_usage(const char *prog_name) {
   std::cerr << "Usage: " << prog_name << " [N] [NUM_TASKS] [DEVICE]\n"
             << "  N          : Size of the vectors (default: 1,000,000)\n"
             << "  NUM_TASKS  : Number of tasks to run (default: 50)\n"
             << "  DEVICE     : 'cpu', 'gpu', or 'fpga' (default: 'cpu')\n"
             << "\nExample: " << prog_name << " 16777216 100 fpga\n";
}

int main(int argc, char *argv[]) {
   // ------ Parsing degli argomenti della riga di comando ------
   if (argc > 1 &&
       (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      return 0;
   }

   size_t N = 1000000;
   size_t NUM_TASKS = 50;
   std::string device_type = "cpu";

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
   // /----- Parsing degli argomenti della riga di comando ------

   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n\n";

   // Creazione dei componenti della pipeline
   Emitter emitter(N, NUM_TASKS);

   // Selezione e creazione dell'acceleratore
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga") {
      accelerator = std::make_unique<FpgaAccelerator>();
   } else if (device_type == "gpu") {
      accelerator = std::make_unique<GpuAccelerator>();
   } else if (device_type == "cpu") {
      accelerator = std::make_unique<CpuAccelerator>();
   } else {
      std::cerr << "[ERROR] Invalid device type '" << device_type << "'.\n\n";
      print_usage(argv[0]);
      return -1;
   }

   // Dati per ottenere il conteggio finale dei task processati
   std::promise<size_t> count_promise;
   std::future<size_t> count_future = count_promise.get_future();

   // Creazione del nodo accelerato con l'acceleratore scelto
   ff_node_acc_t accNode(std::move(accelerator), std::move(count_promise));

   // Creazione della pipeline FF a 2 stadi, il cui secondo stadio incapsula una
   // pipeline interna con 3 thread per i 3 stadi della pipeline di calcolo
   // (upload, execute, download).
   ff::ff_Pipe<> pipe(false, &emitter, &accNode);

   std::cout << "[Main] Starting pipeline execution...\n";
   auto t0 = std::chrono::steady_clock::now();

   // Avvio della pipeline e attesa del completamento
   if (pipe.run_and_wait_end() < 0) {
      std::cerr << "[ERROR] Main: Pipeline execution failed.\n";
      return -1;
   }
   auto t1 = std::chrono::steady_clock::now();
   std::cout << "[Main] Pipeline execution finished.\n";

   size_t final_count = count_future.get();

   // Raccolta dei risultati in nanosecondi
   auto ns_elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
   auto ns_computed = accNode.getComputeTime_ns();

   std::cout << "-------------------------------------------\n"
             << "Total time for " << NUM_TASKS << " tasks on " << device_type
             << ":\n"
             << "N=" << N << " elapsed=" << ns_elapsed << " ns"
             << ", computed=" << ns_computed << " ns\n"
             << "-------------------------------------------\n"
             << "Average time per task:\n"
             << "Avg elapsed=" << ns_elapsed / (NUM_TASKS == 0 ? 1 : NUM_TASKS)
             << " ns/task\n"
             << "Avg computed="
             << ns_computed / (NUM_TASKS == 0 ? 1 : NUM_TASKS) << " ns/task\n"
             << "-------------------------------------------\n"
             << "Tasks processed: " << final_count << " / " << NUM_TASKS
             << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
             << "-------------------------------------------\n";

   // La distruzione degli oggetti (pipe, accNode, accelerator, emitter)
   // viene gestita automaticamente dallo stack.
   return 0;
}
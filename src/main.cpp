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
 * @class Emitter
 * @brief Nodo sorgente della pipeline FastFlow.
 *
 * Emitter genera i Task da processare.
 * Inizializza i dati di input una sola volta e poi crea dinamicamente un nuovo
 * oggetto Task per ogni richiesta dalla pipeline.
 */
class Emitter : public ff::ff_node {
 public:
   /**
    * @param n La dimensione dei vettori da processare.
    * @param num_tasks Il numero totale di task da generare.
    */
   explicit Emitter(size_t n, size_t num_tasks)
       : tasks_to_send(num_tasks), tasks_sent(0) {
      // I vettori di dati vengono allocati e inizializzati una sola volta
      a.resize(n);
      b.resize(n);
      c.resize(n); // Il vettore 'c' viene passato come buffer di output.
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
         // FONDAMENTALE: viene creato un NUOVO oggetto Task sulla heap per ogni
         // invio. Essenziale in un ambiente multi-thread per garantire
         // che ogni task abbia un ciclo di vita indipendente e per evitare
         // race condition. Il nodo successivo sarà responsabile di deallocare
         // questa memoria con 'delete'.
         return new Task{a_ptr_, b_ptr_, c_ptr_, n_};
      }
      // Una volta inviati tutti i task, invia il segnale di fine stream (EOS).
      return FF_EOS;
   }

 private:
   size_t tasks_to_send;
   size_t tasks_sent;
   int *a_ptr_, *b_ptr_, *c_ptr_;
   size_t n_;
   std::vector<int> a, b, c; // I vettori che possiedono i dati.
};

/* --------------- main --------------- */
int main(int argc, char *argv[]) {
   // Parsing degli argomenti da riga di comando con valori di default.
   // Arg 1: Dimensione del vettore (N)
   // Arg 2: Numero di task da eseguire (NUM_TASKS)
   // Arg 3: Tipo di device ('cpu', 'gpu', 'fpga')
   // Default: N=1'000'000, NUM_TASKS=100, device='cpu'
   size_t N = (argc > 1 ? std::stoull(argv[1]) : 1'000'000);
   size_t NUM_TASKS = (argc > 2 ? std::stoull(argv[2]) : 100);
   std::string device_type = (argc > 3 ? argv[3] : "cpu");

   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type << "\n\n";

   // Creazione dei componenti della pipeline
   Emitter emitter(N, NUM_TASKS);

   // Selezione e creazione dell'acceleratore (Pattern Strategy/Factory)
   std::unique_ptr<IAccelerator> accelerator;
   if (device_type == "fpga") {
      accelerator = std::make_unique<FpgaAccelerator>();
   } else if (device_type == "gpu") {
      accelerator = std::make_unique<GpuAccelerator>();
   } else {
      accelerator = std::make_unique<CpuAccelerator>();
   }

   // Verifica che tutti i task siano stati processati
   std::promise<size_t> count_promise;
   std::future<size_t> count_future = count_promise.get_future();

   // Creazione del nodo accelerato con l'acceleratore scelto
   ff_node_acc_t accNode(std::move(accelerator), std::move(count_promise));

   // Composizione e avvio della pipeline
   // La pipeline è composta da 2 stadi: l'Emitter che produce i Task e
   // l'accNode che li consuma, li processa e li conta.
   ff::ff_Pipe<> pipe(false, &emitter, &accNode);

   std::cout << "[Main] Starting pipeline execution...\n";
   auto t0 = std::chrono::steady_clock::now();
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
   auto us_computed = accNode.getComputeTime_ns();

   std::cout << "-------------------------------------------\n"
             << "Total time for " << NUM_TASKS << " tasks on " << device_type
             << ":\n"
             << "N=" << N << " elapsed=" << ns_elapsed << " ns"
             << ", computed=" << us_computed << " ns\n"
             << "-------------------------------------------\n"
             << "Average time per task:\n"
             << "Avg elapsed=" << ns_elapsed / (NUM_TASKS == 0 ? 1 : NUM_TASKS)
             << " ns/task\n"
             << "Avg computed="
             << us_computed / (NUM_TASKS == 0 ? 1 : NUM_TASKS) << " ns/task\n"
             << "-------------------------------------------\n"
             << "Tasks processed: " << final_count << " / " << NUM_TASKS
             << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
             << "-------------------------------------------\n";

   // La distruzione degli oggetti (pipe, accNode, accelerator, emitter)
   // viene gestita automaticamente dallo stack.
   return 0;
}
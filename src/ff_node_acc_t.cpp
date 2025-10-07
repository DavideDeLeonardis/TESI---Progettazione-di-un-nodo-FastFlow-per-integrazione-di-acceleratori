#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <thread>

// Con static SENTINEL ha un valore unico e non nullo
static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

/**
 * Riceve le sue dipendenze (l'acceleratore e la promise)
 */
ff_node_acc_t::ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                             std::promise<size_t> &&count_promise)
    : accelerator_(std::move(acc)), computed_ns_(0), tasks_processed_(0),
      count_promise_(std::move(count_promise)) {}

/**
 * Si occupa di deallocare la memoria delle code interne.
 * Viene chiamato solo dopo che la pipeline è terminata e i thread interni
 * sono stati chiusi in modo sicuro dal metodo svc_end().
 */
ff_node_acc_t::~ff_node_acc_t() {
   delete inQ_;
   delete outQ_;
}

long long ff_node_acc_t::getComputeTime_ns() const {
   return computed_ns_.load();
}

/**
 * @brief Metodo di inizializzazione del nodo, chiamato da FastFlow.
 *
 * Prepara l'intero nodo per l'esecuzione:
 * 1. Inizializza l'acceleratore hardware.
 * 2. Alloca e inizializza le due code interne.
 * 3. Crea e avvia i due thread interni (Producer e Consumer).
 * @return 0 in caso di successo, -1 in caso di errore.
 */
int ff_node_acc_t::svc_init() {
   std::cerr << "[Accelerator Node] Initializing...\n";
   if (!accelerator_ || !accelerator_->initialize()) {
      return -1;
   }

   inQ_ = new TaskQ(1024);
   outQ_ = new ResultQ(1024);
   if (!inQ_->init() || !outQ_->init()) {
      std::cerr << "[ERROR] Accelerator Node: Queues initialization failed.\n";
      return -1;
   }

   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);

   std::cerr << "[Accelerator Node] Internal threads started.\n";
   return 0;
}

/**
 * @brief Metodo principale di input del nodo
 *
 * Riceve i Task dall'Emitter e li inserisce nella coda interna inQ_ per il
 * producerLoop. Non esegue calcoli, ma delega il lavoro, restituendo
 * immediatamente il controllo alla pipeline.
 * @param t Puntatore generico al Task o al segnale FF_EOS.
 * @return FF_GO_ON per continuare, o FF_EOS al termine.
 */
void *ff_node_acc_t::svc(void *t) {
   if (t == FF_EOS) {
      // Se lo stream di input è terminato, inoltra il segnale di terminazione
      // (SENTINEL) ai thread interni.
      if (inQ_)
         inQ_->push(SENTINEL);
      return FF_GO_ON; // La pipeline deve continuare finché i thread non
                       // finiscono.
   }

   auto *task = static_cast<Task *>(t);
   while (!inQ_->push(task))
      std::this_thread::yield();

   return FF_GO_ON;
}

/**
 * @brief Funzione eseguita dal thread "Producer".
 *
 * 1. Attende un Task sulla coda inQ_.
 * 2. Chiama l'acceleratore per eseguire il calcolo (operazione bloccante).
 * 3. Accumula il tempo di calcolo.
 * 4. Crea un oggetto Result e lo inserisce nella coda outQ_.
 * 5. Libera la memoria del Task che ha appena processato.
 */
void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;

   while (true) {
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();

      if (ptr == SENTINEL) {
         // Inoltra il segnale di terminazione al consumer e termina.
         while (!outQ_->push(SENTINEL))
            std::this_thread::yield();
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      long long current_task_ns = 0;

      accelerator_->execute(task, current_task_ns);
      computed_ns_ += current_task_ns;

      auto *res = new Result{task->c, task->n};
      delete task; // Il producer è responsabile di deallocare il Task.
      while (!outQ_->push(res))
         std::this_thread::yield();
   }
}

/**
 * @brief Funzione eseguita dal thread "Consumer".
 *
 * 1. Attende un Result sulla coda outQ_.
 * 2. Incrementa il contatore dei task processati.
 * 3. Libera la memoria dell'oggetto Result.
 * 4. Al termine (ricevendo SENTINEL), mantiene la promessa inviando il
 * conteggio finale al thread main.
 */
void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;

   while (true) {
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      if (ptr == SENTINEL) {
         // Flusso terminato: comunica il conteggio finale al main e termina.
         count_promise_.set_value(tasks_processed_.load());
         break;
      }

      tasks_processed_++;
      auto *res = static_cast<Result *>(ptr);
      delete res; // Il consumer è responsabile di deallocare il Result.
   }
}

/**
 * @brief Metodo di terminazione
 *
 * Gestisce lo spegnimento pulito e sincronizzato dei thread interni.
 * Questo metodo è bloccante e non restituisce il controllo finché entrambi i
 * thread non sono terminati, forzando la pipeline ad attendere.
 */
void ff_node_acc_t::svc_end() {
   std::cerr << "[Accelerator Node] Shutting down internal threads...\n";
   // Invia un'ultima sentinella per garantire lo sblocco dei thread,
   // specialmente se la pipeline si chiude prima che EOS sia stato ricevuto.
   if (inQ_) {
      inQ_->push(SENTINEL);
   }

   // Attende la terminazione dei thread interni.
   if (prodTh_.joinable()) {
      prodTh_.join();
   }
   if (consTh_.joinable()) {
      consTh_.join();
   }

   std::cerr << "[Accelerator Node] Shutdown complete.\n";
}
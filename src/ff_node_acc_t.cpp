#include "ff_node_acc_t.hpp"
#include <chrono>
#include <iostream>
#include <thread>

/**
 * @brief Implementazione del nodo FastFlow che orchestra l'offloading.
 *
 * Questo nodo incapsula una pipeline interna a 3 stadi, gestita da tre thread
 * concorrenti:
 * 1. Uploader: Trasferimento dati asincrono Host -> Device.
 * 2. Launcher: Esecuzione asincrona del kernel sul Device.
 * 3. Downloader: Trasferimento dati sincrono Device -> Host e
 * finalizzazione.
 *
 * Permette di sovrapporre le operazioni di I/O con il calcolo, nella pipeline
 * il task `n` è in esecuzione, mentre i dati per `n+1` vengono caricati e i
 * risultati di `n-1` vengono scaricati.
 */

// Sentinella usata per segnalare la fine dello stream di dati nelle code
// interne ai thread.
static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

/**
 * @brief Costruttore del nodo.

 * @param acc Puntatore a un'implementazione di IAccelerator.
 * @param count_promise Una promise per comunicare il conteggio finale.
 */
ff_node_acc_t::ff_node_acc_t(std::unique_ptr<IAccelerator> acc,
                             std::promise<size_t> &&count_promise)
    : accelerator_(std::move(acc)), count_promise_(std::move(count_promise)) {}

/**
 * @brief Distruttore. Dealloca la memoria per le code interne.
 */
ff_node_acc_t::~ff_node_acc_t() {
   delete inQ_;
   delete kernel_ready_queue_;
   delete readout_ready_queue_;
}

/**
 * @brief Restituisce il tempo di calcolo totale accumulato in nanosecondi.
 */
long long ff_node_acc_t::getComputeTime_ns() const {
   return computed_ns_.load();
}

/**
 * @brief Metodo di inizializzazione del nodo, chiamato da FF.
 */
int ff_node_acc_t::svc_init() {
   std::cerr << "[Accelerator Node] Initializing...\n";
   if (!accelerator_ || !accelerator_->initialize())
      return -1;

   // Alloca le code per la comunicazione tra gli stadi della pipeline interna
   inQ_ = new TaskQ(1024);
   kernel_ready_queue_ = new TaskQ(1024);
   readout_ready_queue_ = new TaskQ(1024);
   if (!inQ_->init() || !kernel_ready_queue_->init() ||
       !readout_ready_queue_->init()) {
      std::cerr << "[ERROR] Accelerator Node: Queues initialization failed.\n";
      return -1;
   }

   // Avvia i tre thread
   uploaderTh_ = std::thread(&ff_node_acc_t::uploaderLoop, this);
   launcherTh_ = std::thread(&ff_node_acc_t::launcherLoop, this);
   downloaderTh_ = std::thread(&ff_node_acc_t::downloaderLoop, this);

   std::cerr << "[Accelerator Node] Internal 3-stage pipeline started.\n\n";
   return 0;
}

/**
 * @brief Metodo principale del nodo, chiamato da FF per ogni task. Riceve un
 * task e lo inserisce nella prima coda (inQ_).
 * @param task Puntatore al task in arrivo.
 * @return FF_GO_ON -> il nodo è pronto a ricevere altri task.
 */
void *ff_node_acc_t::svc(void *task) {
   // Se il task è un EOS, propaga la sentinella alla pipeline interna.
   if (task == FF_EOS) {
      inQ_->push(SENTINEL);
      return FF_GO_ON;
   }

   // Inserisce il task nella coda di ingresso aspettando attivamente se è
   // piena.
   while (!inQ_->push(task))
      std::this_thread::yield();

   return FF_GO_ON;
}

/**
 * @brief Loop per il 1° stadio della pipeline: Uploader.
 * Consumer per inQ_ e producer per kernel_ready_queue_
 */
void ff_node_acc_t::uploaderLoop() {
   void *ptr = nullptr;
   while (true) {
      // Estrae un task dalla coda di ingresso.
      while (!inQ_->pop(&ptr))
         std::this_thread::yield();

      if (ptr == SENTINEL) {
         // Propaga il segnale di terminazione allo stadio successivo.
         kernel_ready_queue_->push(SENTINEL);
         break;
      }

      // Acquisisce un set di buffer libero dal device e gli invia i dati.
      auto *task = static_cast<Task *>(ptr);
      task->buffer_idx = accelerator_->acquire_buffer_set();
      accelerator_->send_data_async(task);

      // Inoltra il task al secondo stadio
      while (!kernel_ready_queue_->push(task))
         std::this_thread::yield();
   }
}

/**
 * @brief Loop per il 2° stadio della pipeline: Launcher.
 * Consumer per kernel_ready_queue_ e producer per readout_ready_queue_
 */
void ff_node_acc_t::launcherLoop() {
   void *ptr = nullptr;
   while (true) {
      // Estrae un task che ha completato il trasferimento dati.
      while (!kernel_ready_queue_->pop(&ptr))
         std::this_thread::yield();

      if (ptr == SENTINEL) {
         // Propaga il segnale di terminazione.
         readout_ready_queue_->push(SENTINEL);
         break;
      }

      // Avvia l'esecuzione del kernel.
      auto *task = static_cast<Task *>(ptr);
      accelerator_->execute_kernel_async(task);

      // Inoltra il task al terzo stadio.
      while (!readout_ready_queue_->push(task))
         std::this_thread::yield();
   }
}

/**
 * @brief Loop del thread per il 3° stadio della pipeline: Downloader.
 * Consumer per readout_ready_queue_.
 */
void ff_node_acc_t::downloaderLoop() {
   void *ptr = nullptr;
   while (true) {
      // Estrae un task.
      while (!readout_ready_queue_->pop(&ptr))
         std::this_thread::yield();

      if (ptr == SENTINEL) {
         // La pipeline è vuota e lo stream è terminato. Comunica il conteggio
         // finale dei task processati al thread main tramite la promise.
         count_promise_.set_value(tasks_processed_.load());
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      long long current_task_ns = 0;

      // Attende il completamento del task e recupera i dati. È l'unica
      // operazione bloccante dell'intera pipeline interna.
      accelerator_->get_results_blocking(task, current_task_ns);

      // Aggiorna le statistiche.
      computed_ns_ += current_task_ns;
      tasks_processed_++;

      // Rilascia il buffer e dealloca la memoria del task.
      accelerator_->release_buffer_set(task->buffer_idx);
      delete task;
   }
}

/**
 * @brief Metodo di terminazione, chiamato da FF.
 */
void ff_node_acc_t::svc_end() {
   // Invia una sentinella "di sicurezza".
   if (inQ_)
      inQ_->push(SENTINEL);

   // Attende la terminazione di ogni thread.
   if (uploaderTh_.joinable())
      uploaderTh_.join();
   if (launcherTh_.joinable())
      launcherTh_.join();
   if (downloaderTh_.joinable())
      downloaderTh_.join();

   std::cerr << "\n[Accelerator Node] Shutdown complete.\n";
}

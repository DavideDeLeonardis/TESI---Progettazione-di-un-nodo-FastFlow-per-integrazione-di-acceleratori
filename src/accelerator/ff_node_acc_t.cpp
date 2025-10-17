#include "ff_node_acc_t.hpp"
#include <iostream>

/**
 * @brief Implementazione del nodo FastFlow che orchestra l'offloading.
 *
 * Il nodo incapsula una pipeline interna a 2 stadi, gestita da due thread:
 * 1. Producer (Upload+Launch): Trasferisce i dati dall'host al device e
 *    avvia l'esecuzione del kernel.
 * 2. Consumer (Download): Trasferisce i risultati dal device all'host.
 */

// Sentinella usata per segnalare la fine dello stream di dati alla pipeline
// interna.
static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

/**
 * @brief Costruttore del nodo.
 *
 * @param acc Puntatore a un'implementazione di IAccelerator.
 * @param stats Puntatore all'oggetto per le statistiche finali.
 */
ff_node_acc_t::ff_node_acc_t(IAccelerator *acc, StatsCollector *stats)
    : accelerator_(acc), stats_(stats) {}

ff_node_acc_t::~ff_node_acc_t() = default;

/**
 * @brief Metodo di inizializzazione del nodo
 */
int ff_node_acc_t::svc_init() {
   std::cerr << "[Accelerator Node] Initializing...\n";

   // Trova il tipo di acceleratore, crea il contesto e la coda di comandi
   // OpenCL, legge il sorgente del kernel, lo compila e prepara l'oggetto
   // kernel, inizializza il pool di buffer e la coda degli indici liberi.
   if (!accelerator_ || !accelerator_->initialize()) {
      std::cerr << "[ERROR] Accelerator setup failed.\n";
      return -1;
   }

   // Avvia i due thread.
   producerTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consumerTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);

   std::cerr << "[Accelerator Node] Internal 2-stage pipeline started.\n\n";
   return 0;
}

/**
 * @brief Metodo principale del nodo, chiamato da FF per ogni task.
 * Rice un task, lo inserisce nella inQ_ e ritorna FF_GO_ON per indicare che è
 * pronto a ricevere un altro task.
 */
void *ff_node_acc_t::svc(void *task) {
   // Se il task è un EOS, propaga la sentinella alla pipeline interna.
   if (task == FF_EOS) {
      inQ_.push(SENTINEL);
      return FF_EOS;
   }

   // Ora di arrivo del task nel nodo.
   static_cast<Task *>(task)->arrival_time = std::chrono::steady_clock::now();

   inQ_.push(task);
   return FF_GO_ON;
}

/**
 * @brief Loop per il 1° stadio della pipeline: Producer (Upload + Launch).
 */
void ff_node_acc_t::producerLoop() {
   while (true) {
      // Attende un task dalla coda di input.
      void *ptr = inQ_.pop();

      // Se riceve la sentinella, la propaga e termina.
      if (ptr == SENTINEL) {
         readyQ_.push(SENTINEL);
         break;
      }

      auto *task = static_cast<Task *>(ptr);

      // Acquisisce un buffer set, invia i dati sul device e avvia il kernel.
      task->buffer_idx = accelerator_->acquire_buffer_set();
      accelerator_->send_data_to_device(task);
      accelerator_->execute_kernel(task);

      readyQ_.push(task);
   }
}

/**
 * @brief Loop per il 2° stadio della pipeline: Consumer (Download).
 */
void ff_node_acc_t::consumerLoop() {
   // Memorizza l'ora di completamento del task precedente.
   std::chrono::steady_clock::time_point last_completion_time;
   bool first_task = true;

   while (true) {
      // Prende un task pronto dalla coda.
      void *ptr = readyQ_.pop();

      if (ptr == SENTINEL) {
         // La pipeline è vuota. Comunica il conteggio finale.
         stats_->count_promise.set_value(stats_->tasks_processed.load());
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      long long current_task_ns = 0;

      // Attende il completamento del kernel e scarica i risultati sull'host.
      accelerator_->get_results_from_device(task, current_task_ns);

      auto end_time = std::chrono::steady_clock::now();

      // Calcola il tempo nel nodo per questo task.
      auto inNode_duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - task->arrival_time);

      // Calcola il tempo dall'ultimo completamento.
      if (!first_task) {
         auto inter_completion_duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
               end_time - last_completion_time);
         stats_->inter_completion_time_ns += inter_completion_duration.count();
      } else {
         first_task = false;
      }
      last_completion_time = end_time;

      // Aggiorna le statistiche sull'oggetto esterno.
      stats_->computed_ns += current_task_ns;
      stats_->ns_total_InNode_time += inNode_duration.count();
      stats_->tasks_processed++;

      accelerator_->release_buffer_set(task->buffer_idx);
      delete task;
   }
}

/**
 * @brief Metodo di terminazione, chiamato da FF. Invia la sentinella ai
 * thread interni e attende la loro terminazione.
 */
void ff_node_acc_t::svc_end() {
   inQ_.push(SENTINEL);

   if (producerTh_.joinable())
      producerTh_.join();
   if (consumerTh_.joinable())
      consumerTh_.join();

   std::cerr << "\n[Accelerator Node] Shutdown complete.\n";
}

#include "CpuAccelerator.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

CpuAccelerator::CpuAccelerator() { std::cerr << "[CpuAccelerator] Created.\n"; }

CpuAccelerator::~CpuAccelerator() {
   std::cerr << "[CpuAccelerator] Destroyed.\n";
}

bool CpuAccelerator::initialize() {
   std::cerr << "[CpuAccelerator] Initialization successful.\n";
   return true;
}

size_t CpuAccelerator::acquire_buffer_set() {
   // La CPU non ha buffer sul device
   return 0;
}

void CpuAccelerator::release_buffer_set(size_t /*index*/) {
   // La CPU non ha buffer da rilasciare
}

void CpuAccelerator::send_data_async(void * /*task_context*/) {
   // I dati sono già nella RAM accessibile alla CPU
}

void CpuAccelerator::execute_kernel_async(void * /*task_context*/) {
   // L'esecuzione avverrà in modo sincrono nel passo successivo
}

/**
 * @brief Esegue il calcolo e attende il risultato.
 */
void CpuAccelerator::get_results_blocking(void *task_context,
                                          long long &computed_ns) {
   auto *task = static_cast<Task *>(task_context);
   std::cerr << "[CpuAccelerator - START] Processing task " << task->id
             << " with N=" << task->n << "...\n";

   auto t0 = std::chrono::steady_clock::now();

   // Esegue la somma vettoriale. Questa funzione è ottimizzata e può sfruttare
   // le istruzioni SIMD della CPU.
   std::transform(task->a,
                  task->a + task->n, // Fine del primo vettore di input
                  task->b, task->c, [](int x, int y) { return x + y; });

   // Calcola il tempo impiegato
   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[CpuAccelerator - END] Task " << task->id << " finished.\n";
}
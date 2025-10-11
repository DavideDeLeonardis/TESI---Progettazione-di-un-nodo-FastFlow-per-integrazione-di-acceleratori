#include "CpuParallelRunner.hpp"
#include "../include/ff_includes.hpp"
#include <chrono>
#include <iostream>
#include <vector>

/**
 * @brief Esegue i task di somma vettoriale in parallelo su tutti i core
 * della CPU utilizzando FastFlow.
 */
long long executeCpuParallelTasks(size_t N, size_t NUM_TASKS) {
   std::cout << "[CPU Parallel] Running tasks in PARALLEL CPU mode...\n\n";

   // Inizializzazione dei dati.
   std::vector<int> a(N), b(N), c(N);
   for (size_t i = 0; i < N; ++i) {
      a[i] = int(i);
      b[i] = int(2 * i);
   }

   // Creazione di ff_parallel_for che gestisce il parallelismo a dati su CPU.
   ParallelFor pf;

   auto t0 = std::chrono::steady_clock::now();

   // Esegue NUM_TASKS volte il calcolo parallelo.
   for (size_t task_num = 0; task_num < NUM_TASKS; ++task_num) {
      std::cerr << "[CPU Parallel - START] Processing task " << task_num + 1
                << " with N=" << N << "...\n";

      // Parallelizza il calcolo della somma vettoriale.
      pf.parallel_for(0, N, 1, 0, [&](const long i) { c[i] = a[i] + b[i]; });

      std::cerr << "[CPU Parallel - END] Task " << task_num + 1
                << " finished.\n";
   }

   auto t1 = std::chrono::steady_clock::now();
   // Return il tempo totale trascorso in nanosecondi
   return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}
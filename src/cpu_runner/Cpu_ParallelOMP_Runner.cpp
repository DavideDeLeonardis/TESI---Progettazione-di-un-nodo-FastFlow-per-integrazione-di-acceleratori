#include "Cpu_ParallelOMP_Runner.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

/**
 * @brief Esegue i task di somma vettoriale in parallelo su tutti i core
 * della CPU utilizzando le direttive OpenMP.
 */
long long executeCpuOMPTasks(size_t N, size_t NUM_TASKS, size_t &tasks_completed) {
   std::cout << "[CPU OpenMP] Running vecAdd tasks in PARALLEL on CPU with OpenMP.\n\n";

   // Inizializzazione dei dati.
   std::vector<int> a(N), b(N), c(N);
   for (size_t i = 0; i < N; ++i) {
      a[i] = int(i);
      b[i] = int(2 * i);
   }

   tasks_completed = 0;
   auto t0 = std::chrono::steady_clock::now();

   // Esegue NUM_TASKS volte il calcolo parallelo.
   for (size_t task_num = 0; task_num < NUM_TASKS; ++task_num) {
      std::cerr << "[CPU OpenMP - START] Processing task " << task_num + 1 << " with N=" << N
                << "...\n";

// Dice al compilatore di parallelizzare il ciclo for distribuendolo tra i thread disponibili.
#pragma omp parallel for
      for (long i = 0; i < N; ++i)
         c[i] = a[i] + b[i];

      std::cerr << "[CPU OpenMP - END] Task " << task_num + 1 << " finished.\n";
      tasks_completed++;
   }

   // Calcola il tempo totale di esecuzione e lo ritorna.
   auto t1 = std::chrono::steady_clock::now();
   return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}
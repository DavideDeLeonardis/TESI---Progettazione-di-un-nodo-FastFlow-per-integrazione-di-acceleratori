#include "Cpu_FF_Runner.hpp"
#include <chrono>
#include <iostream>
#include <vector>

/**
 * @brief Esegue i task di un calcolo specificato da command line in parallelo su tutti i core della
 * CPU utilizzando FastFlow parallel_for.
 */
long long executeCpu_FF_Tasks(size_t N, size_t NUM_TASKS, const std::string &kernel_name,
                              size_t &tasks_completed) {

   // Controlliamo il nome del kernel.
   if (kernel_name != "vecAdd" && kernel_name != "polynomial_op" &&
       kernel_name != "heavy_compute_kernel") {
      std::cerr
         << "[ERROR] CPU Parallel FF: Unknown kernel name '" << kernel_name << "'.\n"
         << "    --> Supported kernels are: 'vecAdd', 'polynomial_op', 'heavy_compute_kernel'.\n";
      exit(EXIT_FAILURE);
   }

   std::cout << "[CPU Parallel FF] Running polynomial operation tasks in PARALLEL on CPU with "
                "FastFlow.\n\n";

   // Inizializzazione dei dati.
   std::vector<int> a(N), b(N), c(N);
   for (size_t i = 0; i < N; ++i) {
      a[i] = int(i);
      b[i] = int(2 * i);
   }

   ParallelFor pf;
   tasks_completed = 0;
   auto t0 = std::chrono::steady_clock::now();

   // Esegue NUM_TASKS volte il calcolo parallelo.
   for (size_t task_num = 0; task_num < NUM_TASKS; ++task_num) {
      std::cerr << "[CPU Parallel FF - START] Processing task " << task_num + 1 << " with N=" << N
                << "...\n";

      // Parallelizza il calcolo usando dell'operazione polinomiale usando ff_parallel_for
      // che gestisce il parallelismo a dati su CPU.
      pf.parallel_for(0, N, 1, 0, [&](const long i) {
         if (kernel_name == "vecAdd") {
            // --------------------------------------------------------------
            // SOMMA VETTORIALE
            // --------------------------------------------------------------
            c[i] = a[i] + b[i];

         } else if (kernel_name == "polynomial_op") {
            // --------------------------------------------------------------
            // OPERAZIONE POLINOMIALE (Calcolo 2a² + 3a³ - 4b² + 5b⁵)
            // --------------------------------------------------------------
            long long val_a = a[i];
            long long val_b = b[i];

            long long a2 = val_a * val_a;
            long long a3 = a2 * val_a;
            long long b2 = val_b * val_b;
            long long b4 = b2 * b2;
            long long b5 = b4 * val_b;

            long long result = (2 * a2) + (3 * a3) - (4 * b2) + (5 * b5);
            c[i] = (int)result;

         } else if (kernel_name == "heavy_compute_kernel") {
            // --------------------------------------------------------------
            // COMPUTAZIONE MOLTO PESANTE (for interno e fz. trigonometriche)
            // --------------------------------------------------------------
            double val_a = (double)a[i];
            double val_b = (double)b[i];
            double result = 0.0;

            for (int j = 0; j < 100; ++j)
               result += std::sin(val_a + j) * std::cos(val_b - j);

            c[i] = (int)result;
         }
      });

      std::cerr << "[CPU Parallel FF - END] Task " << task_num + 1 << " finished.\n";

      tasks_completed++;
   }

   // Ritorna il tempo totale di esecuzione dal primo all'ultimo task.
   auto t1 = std::chrono::steady_clock::now();
   return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}
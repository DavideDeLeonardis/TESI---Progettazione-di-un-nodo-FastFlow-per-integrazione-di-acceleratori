#include "CpuAccelerator.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

CpuAccelerator::CpuAccelerator() { 
   std::cerr << "[CpuAccelerator] Created.\n"; 
}

CpuAccelerator::~CpuAccelerator() {
   std::cerr << "[CpuAccelerator] Destroyed.\n";
}

bool CpuAccelerator::initialize() {
   std::cerr << "[CpuAccelerator] Initializing...\n";
   return true;
}

/**
 * @brief Esegue il calcolo della somma vettoriale direttamente sulla CPU.
 *
 * Utilizza la funzione std::transform della libreria standard C++
 * @param generic_task Puntatore generico al task da eseguire.
 * @param computed_ns Riferimento per restituire il tempo di calcolo misurato.
 */
void CpuAccelerator::execute(void *generic_task, long long &computed_ns) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "\n[CpuAccelerator - START] Executing task with N=" << task->n << "...\n";

   auto t0 = std::chrono::steady_clock::now();

   // Esegue la somma vettoriale
   // std::transform è altamente ottimizzato e può sfruttare le istruzioni
   // vettoriali (SIMD) della CPU per accelerare il calcolo.
   std::transform(
      task->a,           // Inizio del primo vettore di input
      task->a + task->n, // Fine del primo vettore di input
      task->b,           // Inizio del secondo vettore di input
      task->c,           // Inizio del vettore di output
      [](int x, int y) { return x + y; }); // Somma

   auto t1 = std::chrono::steady_clock::now();
   computed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

   std::cerr << "[CpuAccelerator - END] Task execution finished.\n";
}
#include "CpuAccelerator.hpp"
#include <algorithm> // Necessario per std::transform
#include <chrono>
#include <iostream>

CpuAccelerator::CpuAccelerator() { std::cerr << "[CpuAccelerator] Created.\n"; }

CpuAccelerator::~CpuAccelerator() {
   std::cerr << "[CpuAccelerator] Destroyed.\n";
}

// L'inizializzazione per la CPU è banale: non c'è nulla da fare.
bool CpuAccelerator::initialize() {
   std::cerr << "[CpuAccelerator] Initializing... (nothing to do)\n";
   return true;
}

// Il metodo execute contiene la logica di calcolo originale su CPU.
void CpuAccelerator::execute(void *generic_task, long long &computed_us) {
   auto *task = static_cast<Task *>(generic_task);

   auto t0 = std::chrono::steady_clock::now();

   std::transform(task->a, task->a + task->n, task->b, task->c,
                  [](int x, int y) { return x + y; });

   auto t1 = std::chrono::steady_clock::now();

   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}
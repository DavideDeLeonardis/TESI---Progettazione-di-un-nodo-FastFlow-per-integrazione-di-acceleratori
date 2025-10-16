#pragma once

#include "../../include/ff_includes.hpp"
#include <cstddef>

/**
 * @brief Esegue i task di somma vettoriale in parallelo su tutti i core
 * della CPU utilizzando FastFlow.
 *
 * @param N La dimensione dei vettori per ogni task.
 * @param NUM_TASKS Il numero totale di task da eseguire in sequenza.
 * @param tasks_completed Il numero dei task effettivamente completati.
 * @return ns_elapsed (tempo totale per completare tutti i task in nanosecondi).
 */
long long executeCpuParallelTasks(size_t N, size_t NUM_TASKS,
                                  size_t &tasks_completed);
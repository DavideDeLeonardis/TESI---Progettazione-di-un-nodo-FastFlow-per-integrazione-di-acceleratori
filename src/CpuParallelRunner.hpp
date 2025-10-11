#pragma once
#include <cstddef>

/**
 * @brief Esegue i task di somma vettoriale in parallelo su tutti i core
 * della CPU utilizzando FastFlow.
 *
 * @param N La dimensione dei vettori per ogni task.
 * @param NUM_TASKS Il numero totale di task da eseguire in sequenza.
 * @return Il tempo totale trascorso in nanosecondi.
 */
long long executeCpuParallelTasks(size_t N, size_t NUM_TASKS);
#pragma once

#include <cstddef>

/**
 * @brief Esegue i task di somma vettoriale in parallelo su tutti i core
 * della CPU utilizzando le direttive OpenMP.
 * @param N La dimensione dei vettori per ogni task.
 * @param NUM_TASKS Il numero totale di task da eseguire.
 * @param tasks_completed Riferimento per memorizzare il numero di task completati.
 * @return long long Il tempo totale trascorso in nanosecondi.
 */
long long executeCpuOMPTasks(size_t N, size_t NUM_TASKS, size_t &tasks_completed);
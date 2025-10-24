#pragma once

#include "../../include/ff_includes.hpp"
#include <cstddef>

/**
 * @brief Esegue i task di un'operazione polinomiale complessa (2a² + 3a³ - 4b² + 5b⁵) in parallelo
 * su tutti i core della CPU utilizzando FastFlow parallel_for.
 *
 * @param N La dimensione dei vettori per ogni task.
 * @param NUM_TASKS Il numero totale di task da eseguire in sequenza.
 * @param kernel_name Il nome del kernel da eseguire ("vecAdd", "polynomial_op" o
 * "heavy_compute_kernel").
 * @param tasks_completed Il numero dei task effettivamente completati.
 * @return elapsed_ns (tempo totale per completare tutti i task in nanosecondi).
 */
long long executeCpu_FF_Tasks(size_t N, size_t NUM_TASKS, const std::string &kernel_name,
                              size_t &tasks_completed);
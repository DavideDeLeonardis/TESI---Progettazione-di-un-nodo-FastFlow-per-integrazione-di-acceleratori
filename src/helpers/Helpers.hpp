#pragma once

#include <cstddef>
#include <string>

// Helper per stampare le istruzioni d'uso.
void print_usage(const char *prog_name);

// Helper per stampare le statistiche finali.
void print_stats(size_t N, size_t NUM_TASKS, const std::string &device_type,
                 long long ns_elapsed, long long ns_computed,
                 size_t final_count);
#pragma once

#include <cstddef>
#include <string>

// Helper per il parsing degli argomenti della riga di comando.
void parse_args(int argc, char *argv[], size_t &N, size_t &NUM_TASKS,
                std::string &device_type);

// Helper per stampare le istruzioni d'uso.
void print_usage(const char *prog_name);

// Helper per stampare le statistiche finali.
void print_stats(size_t N, size_t NUM_TASKS, const std::string &device_type,
                 long long ns_elapsed, long long ns_computed,
                 size_t final_count);
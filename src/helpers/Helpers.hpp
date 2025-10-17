#pragma once

#include <cstddef>
#include <string>

// Helpers per stampare la configurazione attuale della computazione.
void print_configuration(size_t N, size_t NUM_TASKS,
                         const std::string &device_type,
                         const std::string &kernel_path);

// Helper per il parsing degli argomenti della riga di comando.
void parse_args(int argc, char *argv[], size_t &N, size_t &NUM_TASKS,
                std::string &device_type, std::string &kernel_path,
                std::string &kernel_name);

// Helper per stampare le istruzioni d'uso.
void print_usage(const char *prog_name);

// Helper per stampare le statistiche finali.
void print_stats(size_t N, size_t NUM_TASKS, const std::string &device_type,
                 std::string &kernel_name, long long ns_elapsed,
                 long long ns_computed, long long ns_total_service_time,
                 long long ns_inter_completion_time, size_t final_count);

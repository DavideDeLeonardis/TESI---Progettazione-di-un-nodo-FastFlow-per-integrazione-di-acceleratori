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

// Helper per calcolare e stampare le metriche finali.
void calculate_and_print_metrics(size_t N, size_t NUM_TASKS,
                                 const std::string &device_type,
                                 std::string &kernel_name, long long elapsed_ns,
                                 long long computed_ns,
                                 long long total_InNode_time_ns,
                                 long long inter_completion_time_ns,
                                 size_t final_count);

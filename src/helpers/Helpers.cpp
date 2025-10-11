#include "Helpers.hpp"
#include <iostream>

// Helper per stampare le istruzioni d'uso.
void print_usage(const char *prog_name) {
   std::cerr << "Usage: " << prog_name << " [N] [NUM_TASKS] [DEVICE]\n"
             << "  N          : Size of the vectors (default: 1,000,000)\n"
             << "  NUM_TASKS  : Number of tasks to run (default: 50)\n"
             << "  DEVICE     : 'cpu', 'gpu', or 'fpga' (default: 'cpu')\n"
             << "\nExample: " << prog_name << " 16777216 100 gpu\n";
}

// Helper per stampare le statistiche finali.
void print_stats(size_t N, size_t NUM_TASKS, const std::string &device_type,
                 long long ns_elapsed, long long ns_computed,
                 size_t final_count) {
   std::cout << "-------------------------------------------\n"
             << "Total time for " << NUM_TASKS << " tasks on " << device_type
             << ":\n"
             << "N=" << N << " elapsed=" << ns_elapsed << " ns"
             << ", computed=" << ns_computed << " ns\n"
             << "-------------------------------------------\n"
             << "Average time per task:\n"
             << "Avg elapsed=" << ns_elapsed / (NUM_TASKS == 0 ? 1 : NUM_TASKS)
             << " ns/task\n"
             << "Avg computed="
             << ns_computed / (NUM_TASKS == 0 ? 1 : NUM_TASKS) << " ns/task\n"
             << "-------------------------------------------\n"
             << "Tasks processed: " << final_count << " / " << NUM_TASKS
             << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
             << "-------------------------------------------\n";
}
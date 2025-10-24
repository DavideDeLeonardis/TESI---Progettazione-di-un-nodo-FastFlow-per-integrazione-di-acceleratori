#include "Helpers.hpp"
#include <algorithm>
#include <iostream>

/**
 * Helper interno per estrarre il nome del file da un percorso, senza
 * estensione. Usata per estrapolare il nome della funzione kernel dal kernel
 * file.
 */
static std::string extractKernelName(const std::string &path) {
   if (path.empty())
      return "";

   size_t last_slash_pos = path.find_last_of("/\\");
   std::string filename =
      (last_slash_pos == std::string::npos) ? path : path.substr(last_slash_pos + 1);

   size_t dot_pos = filename.find_first_of('.');
   if (dot_pos == std::string::npos)
      return filename;

   return filename.substr(0, dot_pos);
}

/**
 * Helper per il parsing degli argomenti della riga di comando.
 */
void parse_args(int argc, char *argv[], size_t &N, size_t &NUM_TASKS, std::string &device_type,
                std::string &kernel_path, std::string &kernel_name) {
   if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
      print_usage(argv[0]);
      exit(0);
   }

   if (argc > 5)
      std::cerr << "[WARNING] Too many arguments provided. Ignoring extras.\n";

   try {
      if (argc > 1)
         N = std::stoull(argv[1]);
      if (argc > 2)
         NUM_TASKS = std::stoull(argv[2]);
      if (argc > 3)
         device_type = argv[3];
      if (argc > 4)
         kernel_path = argv[4];
   } catch (const std::invalid_argument &e) {
      std::cerr << "[ERROR] Invalid numeric argument provided.\n\n";
      print_usage(argv[0]);
      exit(-1);
   }

   if (N == 0 || NUM_TASKS == 0) {
      std::cerr << "\n[FATAL] La dimensione dei vettori (N) o del numero dei task (NUM_TASKS) non "
                   "può essere 0.\n";
      exit(EXIT_FAILURE);
   }

   // Per GPU e FPGA, se non specifico un kernel di default imposta polynomial_op.
   if (device_type == "gpu_opencl" && kernel_path.empty())
      kernel_path = "kernels/gpu/polynomial_op.cl";
   else if (device_type == "gpu_metal" && kernel_path.empty())
      kernel_path = "kernels/gpu/polynomial_op.metal";
   else if (device_type == "fpga" && kernel_path.empty())
      kernel_path = "kernels/fpga/krnl_polynomial_op.xclbin";

   // Per GPU e FPGA, estraggo il nome del kernel dal percorso specificato.
   if (device_type == "gpu_opencl" || device_type == "fpga" || device_type == "gpu_metal")
      kernel_name = extractKernelName(kernel_path);

   // Per CPU, se non specifico un kernel imposta polynomial_op, altrimenti lo estrae dal nome.
   if (kernel_path.empty() && (device_type == "cpu_ff" || device_type == "cpu_omp"))
      kernel_name = "polynomial_op";
   else
      kernel_name = extractKernelName(kernel_path);
}

/**
 * Helper per stampare la configurazione di esecuzione del programma.
 */
void print_configuration(size_t N, size_t NUM_TASKS, const std::string &device_type,
                         const std::string &kernel_path, const std::string &kernel_name) {
   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type;

   if (device_type == "cpu_ff" || device_type == "cpu_omp")
      std::cout << ", Kernel=" << kernel_name;

   if (device_type == "gpu_opencl" || device_type == "gpu_metal" || device_type == "fpga")
      std::cout << ", Using " << kernel_path;

   std::cout << "\n\n";
}

/**
 * Helper per stampare le istruzioni d'uso.
 */
void print_usage(const char *prog_name) {
   std::cerr << "Usage: " << prog_name << " [N] [NUM_TASKS] [DEVICE] [KERNEL_PATH]\n"
             << "  N            : Size of the vectors (default: 1,000,000)\n"
             << "  NUM_TASKS    : Number of tasks to run (default: 20)\n"
             << "  DEVICE       : 'cpu_ff', 'cpu_omp', 'gpu_opencl', 'gpu_metal' or 'fpga' "
                "(default: 'cpu_ff').\n"
             << "  KERNEL_PATH  : Path to the kernel file for accelerators (.cl, .xclbin, .metal)\n"
             << "                 or kernel name for CPU ('vecAdd', 'polynomial_op', etc.)\n"
             << "\nExample (GPU): " << prog_name
             << " 16777216 100 gpu_opencl kernels/gpu/heavy_compute_kernel.cl\n"
             << "Example (CPU): " << prog_name << " 16777216 100 cpu_ff vecAdd\n";
}

/**
 * @brief Calcola le metriche di performance finali a partire dai dati grezzi.
 */
PerformanceData calculate_metrics(long long elapsed_ns, long long computed_ns,
                                  long long total_InNode_time_ns,
                                  long long inter_completion_time_ns, size_t final_count) {

   PerformanceData metrics;
   if (final_count == 0)
      return metrics;

   // Tempo medio tra il completamento di due task consecutivi (in ms).
   if (final_count > 1)
      metrics.avg_service_time_ms = (inter_completion_time_ns / (final_count - 1)) / 1.0e6;

   // Tempo totale che la pipeline impiega per processare tutti i task (in sec).
   metrics.elapsed_s = elapsed_ns / 1.0e9;
   // Tempo medio per un task dall'ingresso all'uscita del nodo (in ms).
   metrics.avg_InNode_time_ms = (total_InNode_time_ns / final_count) / 1.0e6;
   // Tempo medio del singolo calcolo sull'acceleratore, senza overhead (in ms).
   metrics.avg_computed_ms = (computed_ns / final_count) / 1.0e6;
   // Costo medio di gestione: trasferimento dati, uso delle code, etc.
   metrics.avg_overhead_ms = metrics.avg_InNode_time_ms - metrics.avg_computed_ms;
   // Task totali processati al secondo.
   metrics.throughput = (metrics.elapsed_s > 0) ? (final_count / metrics.elapsed_s) : 0;

   return metrics;
}

/**
 * Helper per calcolare e stampare le statistiche finali.
 */
void print_metrics(size_t N, size_t NUM_TASKS, const std::string &device_type,
                   const std::string &kernel_name, const PerformanceData &metrics,
                   size_t final_count) {

   if (final_count == 0) {
      std::cout << "-----------------------------------------------\n"
                << "No tasks were processed. No metrics to display.\n"
                << "-----------------------------------------------\n";
      return;
   }

   // Trasforma in uppercase il device_type.
   std::string DEVICE_TYPE = device_type;
   std::transform(DEVICE_TYPE.begin(), DEVICE_TYPE.end(), DEVICE_TYPE.begin(),
                  [](unsigned char c) { return std::toupper(c); });

   std::cout << "\n------------------------------------------------------------"
                "------\n"
             << "PERFORMANCE METRICS on " << DEVICE_TYPE << "\n   (N=" << N
             << ", Tasks=" << final_count;

   if (device_type == "cpu_ff" || device_type == "cpu_omp") {
      // Tempo medio per completare un singolo task.
      double avg_task_time_ms = metrics.elapsed_s * 1000 / final_count;

      std::cout << ", Kernel=" << kernel_name
                << ")\n------------------------------------------------------------------\n"
                << "Avg Time per Task: " << avg_task_time_ms << " ms/task\n"
                << "   (Tempo medio per completare un singolo task in modo sequenziale)\n\n"
                << "Throughput: " << metrics.throughput << " tasks/sec\n"
                << "   (Task totali processati al secondo)\n\n"
                << "Total Time Elapsed: " << metrics.elapsed_s << " s\n"
                << "------------------------------------------------------------------\n"
                << "Tasks processed: " << final_count << " / " << NUM_TASKS
                << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
                << "------------------------------------------------------------------\n";

   } else
      std::cout << ", Kernel=" << kernel_name
                << ")\n------------------------------------------------------------------"
                   "\n"
                << "Avg Service Time: " << metrics.avg_service_time_ms << " ms/task\n"
                << "   (Tempo medio tra il completamento di due task consecutivi)\n\n"
                << "Avg In_Node Time: " << metrics.avg_InNode_time_ms << " ms/task\n"
                << "   (Tempo medio per un task dall'ingresso all'uscita del nodo)\n\n"
                << "Avg Pure Compute Time: " << metrics.avg_computed_ms << " ms/task\n"
                << "   (Tempo medio di un singolo calcolo sull'acceleratore, senza "
                   "overhead)\n\n"
                << "Avg Overhead Time: " << metrics.avg_overhead_ms << " ms/task\n"
                << "   (Costo medio di gestione: trasferimento dati, uso delle code, etc.)\n\n"
                << "Throughput: " << metrics.throughput << " tasks/sec\n"
                << "   (Task totali processati al secondo)\n\n"
                << "Total Time Elapsed: " << metrics.elapsed_s << " s\n"
                << "------------------------------------------------------------------\n"
                << "Tasks processed: " << final_count << " / " << NUM_TASKS
                << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
                << "------------------------------------------------------------------\n";
}

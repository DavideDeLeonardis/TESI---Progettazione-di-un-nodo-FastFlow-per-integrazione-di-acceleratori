#include "Helpers.hpp"
#include <iostream>

// Helper interno per estrarre il nome del file da un percorso, senza
// estensione. Usata per estrapolare il nome della funzione kernel dal kernel
// file.
static std::string extractKernelName(const std::string &path) {
   if (path.empty())
      return "";

   size_t last_slash_pos = path.find_last_of("/\\");
   std::string filename = (last_slash_pos == std::string::npos)
                             ? path
                             : path.substr(last_slash_pos + 1);

   size_t dot_pos = filename.find_first_of('.');
   if (dot_pos == std::string::npos)
      return filename;

   return filename.substr(0, dot_pos);
}

// Helper per il parsing degli argomenti della riga di comando.
void parse_args(int argc, char *argv[], size_t &N, size_t &NUM_TASKS,
                std::string &device_type, std::string &kernel_path,
                std::string &kernel_name) {
   if (argc > 1 &&
       (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
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

   if (N == 0) {
      std::cerr << "[FATAL] La dimensione dei vettori (N) non può essere 0.\n";
      exit(EXIT_FAILURE);
   }

   // Se gpu o fpga, e il percorso non è stato specificato, imposta quello di
   // default.
   if (device_type == "gpu") {
      if (kernel_path.empty())
         kernel_path = "kernels/gpu/polynomial_op.cl";
   } else if (device_type == "fpga") {
      if (kernel_path.empty())
         kernel_path = "kernels/fpga/krnl_polynomial_op.xclbin";
   }

   // Estrae il nome del kernel dal percorso del file, se GPU o FPGA.
   if (device_type == "gpu" || device_type == "fpga")
      kernel_name = extractKernelName(kernel_path);
}

// Helper per stampare la configurazione di esecuzione del programma.
void print_configuration(size_t N, size_t NUM_TASKS,
                         const std::string &device_type,
                         const std::string &kernel_path) {
   std::cout << "\nConfiguration: N=" << N << ", NUM_TASKS=" << NUM_TASKS
             << ", Device=" << device_type;

   if (device_type == "gpu" || device_type == "fpga")
      std::cout << ", Using " << kernel_path;

   std::cout << "\n\n";
}

// Helper per stampare le istruzioni d'uso.
void print_usage(const char *prog_name) {
   std::cerr << "Usage: " << prog_name
             << " [N] [NUM_TASKS] [DEVICE] [KERNEL_PATH]\n"
             << "  N            : Size of the vectors (default: 1,000,000)\n"
             << "  NUM_TASKS    : Number of tasks to run (default: 50)\n"
             << "  DEVICE       : 'cpu', 'gpu', or 'fpga' (default: 'cpu')\n"
             << "  KERNEL_PATH  : If on GPU or FPGA, path to the kernel file "
                "(.cl or .xclbin)\n"
             << "\nExample: " << prog_name
             << " 16777216 100 gpu kernels/polynomial_op.cl\n";
}

// Helper per stampare le statistiche finali.
void print_stats(size_t N, size_t NUM_TASKS, const std::string &device_type,
                 std::string &kernel_name, long long ns_elapsed,
                 long long ns_computed, long long ns_total_service_time,
                 long long ns_inter_completion_time, size_t final_count) {

   double elapsed_s = ns_elapsed / 1.0e9; // Trasformato in sec
   double avg_service_time_ms =
      (ns_total_service_time / final_count) / 1.0e6; // Trasformato in ms
   double throughput = (elapsed_s > 0) ? (final_count / elapsed_s) : 0;
   double avg_computed_ms =
      (ns_computed / final_count) / 1.0e6; // Trasformato in ms
   double avg_overhead_ms = avg_service_time_ms - avg_computed_ms;

   double avg_pipeline_period_ms = 0.0;
   if (final_count > 1)
      avg_pipeline_period_ms =
         (ns_inter_completion_time / (final_count - 1)) / 1.0e6;

   std::cout << "\n------------------------------------------------------------"
                "------\n"
             << "PERFORMANCE METRICS on " << device_type << "\nN=" << N
             << ", Tasks=" << final_count;

   if (!kernel_name.empty())
      std::cout << ", Kernel=" << kernel_name;

   std::cout
      << "\n------------------------------------------------------------------\n"
      << "Avg Service Time: " << avg_service_time_ms << " ms/task\n"
      << "   (Tempo medio per un task dall'ingresso all'uscita del nodo)\n\n"
      << "Avg Pipeline Period: " << avg_pipeline_period_ms << " ms/task\n"
      << "   (Tempo medio tra il completamento di due task consecutivi)\n\n"
      << "Throughput: " << throughput << " tasks/sec\n"
      << "   (Task totali processati al secondo)\n\n"
      << "Avg Pure Compute Time: " << avg_computed_ms << " ms/task\n"
      << "   (Tempo medio di calcolo sull'acceleratore, senza overhead)\n\n"
      << "Avg Overhead Time: " << avg_overhead_ms << " ms/task\n"
      << "   (Costo medio di gestione: trasferimento dati, code, etc.)\n\n"
      << "Total Time Elapsed: " << elapsed_s << " s\n"
      << "------------------------------------------------------------------\n"
      << "Tasks processed: " << final_count << " / " << NUM_TASKS
      << (final_count == NUM_TASKS ? " (SUCCESS)" : " (FAILURE)") << "\n"
      << "------------------------------------------------------------------\n";
}

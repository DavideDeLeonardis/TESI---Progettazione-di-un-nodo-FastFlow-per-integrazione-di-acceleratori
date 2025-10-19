#pragma once
#include <cstddef>

/**
 * @brief Struttura per contenere le metriche di performance calcolate.
 */
struct PerformanceData {
   double avg_service_time_ms = 0.0;
   double avg_InNode_time_ms = 0.0;
   double avg_computed_ms = 0.0;
   double avg_overhead_ms = 0.0;
   double throughput = 0.0;
   double elapsed_s = 0.0;
};
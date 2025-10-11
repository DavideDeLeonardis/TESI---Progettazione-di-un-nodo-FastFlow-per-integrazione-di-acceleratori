#pragma once

#include <atomic>
#include <future>

/**
 * @brief Oggetto per raccogliere le statistiche dai nodi della pipeline.
 */
struct StatsCollector {
   std::atomic<long long> computed_ns{0};
   std::atomic<size_t> tasks_processed{0};
   std::promise<size_t> count_promise;
};
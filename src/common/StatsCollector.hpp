#pragma once

#include <atomic>
#include <future>

// Struttura usata per raccogliere risultati generati dai thread interni e
// passarli al thread principale.
struct StatsCollector {
   std::atomic<size_t> tasks_processed{0};
   std::promise<size_t> count_promise;
   std::atomic<long long> computed_ns{0};
   std::atomic<long long> total_service_time_ns{0};
   std::atomic<long long> inter_completion_time_ns{0};
};
#pragma once

#include <atomic>
#include <future>

// Struttura usata per raccogliere risultati generati dai thread interni e
// passarli al thread principale.
struct StatsCollector {
   std::atomic<long long> computed_ns{0};
   std::atomic<size_t> tasks_processed{0};
   std::promise<size_t> count_promise;
};
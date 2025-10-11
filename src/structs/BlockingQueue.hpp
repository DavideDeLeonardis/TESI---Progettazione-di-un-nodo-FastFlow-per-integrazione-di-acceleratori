#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

/**
 * @brief Coda bloccante e thread-safe per la comunicazione tra stadi.
 *
 * Mette i thread consumer a dormire quando la coda è vuota e li risveglia
 * quando un nuovo elemento è disponibile, evitando l'attesa attiva.
 */
template <typename T> class BlockingQueue {
 public:
   void push(T value) {
      {
         std::lock_guard<std::mutex> lock(mutex_);
         queue_.push(std::move(value));
      }

      notEmptyCondition_.notify_one();
   }

   T pop() {
      std::unique_lock<std::mutex> lock(mutex_);

      notEmptyCondition_.wait(lock, [this] { return !queue_.empty(); });

      T item = std::move(queue_.front());
      queue_.pop();
      return item;
   }

 private:
   std::queue<T> queue_;
   std::mutex mutex_;
   std::condition_variable notEmptyCondition_;
};
#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads);
  ~ThreadPool();

  // Disallow copying the thread pool.
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void enqueueTask(std::function<void()> task);
  void stopThreadPool();
  void waitForAllTasks();  // Wait for all submitted tasks

 private:
  std::vector<std::thread> workers;             // Worker threads
  std::queue<std::function<void()>> taskQueue;  // Task queue
  std::mutex queueMutex;                        // Task queue mutex
  std::condition_variable condition;            // Condition variable
  std::atomic<bool> stop;                       // Stop flag

  std::atomic<size_t> taskCount;                // Atomic count of outstanding tasks
  std::mutex completionMutex;                   // Task completion mutex
  std::condition_variable completionCondition;  // Task completion condition variable
};

#endif  // THREAD_POOL_HPP

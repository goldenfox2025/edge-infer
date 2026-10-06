#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
 public:
  // Zero workers executes tasks inline on the submitting thread.
  explicit ThreadPool(size_t numThreads);
  ~ThreadPool();

  // Disallow copying the thread pool.
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void enqueueTask(std::function<void()> task);
  // Drain and join from a controlling thread, never from one of this pool's tasks.
  void stopThreadPool();
  // Drain all tasks, then rethrow the first task failure once. The pool remains usable.
  void waitForAllTasks();

 private:
  void executeTask(std::function<void()> task) noexcept;
  const bool inlineMode;
  std::vector<std::thread> workers;             // Worker threads
  std::queue<std::function<void()>> taskQueue;  // Task queue
  std::mutex queueMutex;                        // Task queue mutex
  std::condition_variable condition;            // Condition variable
  bool stop = false;                           // Protected by queueMutex
  size_t taskCount = 0;                         // Queued and executing tasks
  std::exception_ptr firstFailure;
  std::condition_variable completionCondition;  // Task completion condition variable
};

#endif  // THREAD_POOL_HPP

#include "thread_pool.hpp"

ThreadPool::ThreadPool(size_t numThreads)
    : inlineMode(numThreads == 0) {
  try {
    workers.reserve(numThreads);
    for (size_t i = 0; i < numThreads; i++) {
      workers.emplace_back([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this]() { return stop || !taskQueue.empty(); });
            if (stop && taskQueue.empty()) return;
            task = std::move(taskQueue.front());
            taskQueue.pop();
          }
          executeTask(std::move(task));
        }
      });
    }
  } catch (...) {
    // A failed constructor has no destructor to join workers already created.
    stopThreadPool();
    throw;
  }
}

void ThreadPool::executeTask(std::function<void()> task) noexcept {
  std::exception_ptr failure;
  try { task(); }
  catch (...) { failure = std::current_exception(); }
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (failure && !firstFailure) firstFailure = failure;
    --taskCount;
    if (!taskCount) completionCondition.notify_all();
  }
}

void ThreadPool::enqueueTask(std::function<void()> task) {
  if (!task) throw std::invalid_argument("ThreadPool requires a nonempty task");
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
    if (!inlineMode) taskQueue.push(std::move(task));
    ++taskCount;
  }
  if (inlineMode) executeTask(std::move(task));
  else condition.notify_one();
}

void ThreadPool::waitForAllTasks() {
  std::exception_ptr failure;
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    completionCondition.wait(lock, [this]() { return taskCount == 0; });
    failure = std::exchange(firstFailure, {});
  }
  if (failure) std::rethrow_exception(failure);
}

void ThreadPool::stopThreadPool() {
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (stop) return;
    stop = true;
  }
  condition.notify_all();
  for (auto& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  std::unique_lock<std::mutex> lock(queueMutex);
  completionCondition.wait(lock, [this]() { return taskCount == 0; });
}

// ThreadPool destructor
ThreadPool::~ThreadPool() { stopThreadPool(); }

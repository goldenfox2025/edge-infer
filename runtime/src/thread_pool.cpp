#include "thread_pool.hpp"

ThreadPool::ThreadPool(size_t numThreads)
    : stop(false), taskCount(0) {  // Initialize the outstanding task count
  for (size_t i = 0; i < numThreads; i++) {
    workers.emplace_back([this]() {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queueMutex);
          this->condition.wait(lock, [this]() {
            return this->stop || !this->taskQueue.empty();
          });
          if (this->stop && this->taskQueue.empty()) {
            return;
          }

          task = std::move(this->taskQueue.front());
          this->taskQueue.pop();
        }
        task();
        // After execution, decrement the task count and notify completion.
        if (--taskCount == 0) {
          std::unique_lock<std::mutex> lock(completionMutex);
          completionCondition.notify_one();  // Notify task completion
        }
      }
    });
  }
}

// Enqueue a task.
void ThreadPool::enqueueTask(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    taskQueue.push(std::move(task));
    taskCount++;  // Increment the outstanding task count
  }
  condition.notify_one();
}

void ThreadPool::waitForAllTasks() {
  std::unique_lock<std::mutex> lock(completionMutex);
  completionCondition.wait(lock, [this]() { return taskCount == 0; });
}

void ThreadPool::stopThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    if (stop) {
      return;
    }
    stop = true;
  }
  condition.notify_all();
  for (auto& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

// ThreadPool destructor
ThreadPool::~ThreadPool() { stopThreadPool(); }

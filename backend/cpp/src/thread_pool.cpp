#include "thread_pool.hpp"

ThreadPool::ThreadPool(size_t numThreads)
    : stop(false), taskCount(0) {  // 初始化 taskCount 为 0
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
        // 任务执行完成后，减少任务计数并检查是否完成
        if (--taskCount == 0) {
          std::unique_lock<std::mutex> lock(completionMutex);
          completionCondition.notify_one();  // 通知任务完成
        }
      }
    });
  }
}

// 向任务队列添加任务
void ThreadPool::enqueueTask(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    if (stop) {
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }
    taskQueue.push(std::move(task));
    taskCount++;  // 增加任务计数
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

// ThreadPool 的析构函数
ThreadPool::~ThreadPool() { stopThreadPool(); }

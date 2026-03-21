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

  // 禁止拷贝构造和拷贝赋值
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void enqueueTask(std::function<void()> task);
  void stopThreadPool();
  void waitForAllTasks();  // 新增等待所有任务完成的方法

 private:
  std::vector<std::thread> workers;             // 工作线程
  std::queue<std::function<void()>> taskQueue;  // 任务队列
  std::mutex queueMutex;                        // 任务队列锁
  std::condition_variable condition;            // 条件变量
  std::atomic<bool> stop;                       // 停止标志

  std::atomic<size_t> taskCount;                // 原子计数器，记录任务数量
  std::mutex completionMutex;                   // 完成条件变量的互斥锁
  std::condition_variable completionCondition;  // 完成条件变量
};

#endif  // THREAD_POOL_HPP
